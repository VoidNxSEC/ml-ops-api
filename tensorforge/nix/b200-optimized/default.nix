# NVIDIA B200 Optimized TensorForge Configuration
#
# This NixOS module provides hardware-optimized configurations for
# TensorForge running on NVIDIA B200 GPUs with 192GB HBM3e memory.
#
# Features:
# - Tensor parallelism optimized for B200 architecture
# - Memory-aware scheduling for 192GB HBM3e
# - vLLM optimizations for long context (131k tokens)
# - llama.cpp optimizations for maximum GPU utilization
# - System tuning for high-throughput inference pipelines
#
# Usage in configuration.nix:
#   imports = [
#     ./tensorforge/nix/b200-optimized/default.nix
#   ];
#
#   services.tensorforge.b200Optimized = {
#     enable = true;
#     vllm.enable = true;
#     llamacpp.enable = true;
#   };

{
  config,
  lib,
  pkgs,
  ...
}:

with lib;

let
  cfg = config.services.tensorforge.b200Optimized;

  # B200-specific hardware constants
  b200Specs = {
    totalVramGB = 192;
    memoryBandwidthGBs = 8000; # HBM3e
    tensorCores = 576;
    cudaCores = 16896;
    tdpWatts = 1000;
  };

  # Calculate optimal tensor parallelism based on B200 architecture
  # B200 has 4 GPU dies, so tensor_parallel_size = 4 for optimal distribution
  optimalTensorParallelSize = 4;

  # Calculate optimal pipeline parallelism
  # For B200 with 192GB, we can split large models across pipeline stages
  optimalPipelineParallelSize = 2;

  # Maximum model length supported by B200 with optimization
  # B200 can handle very long contexts with optimized attention
  maxModelLength = 131072;

  # GPU memory utilization target (0.0-1.0)
  # Higher utilization for B200 since we have plenty of VRAM
  gpuMemoryUtilization = 0.95;

  # B200-optimized vLLM service
  vllmService = pkgs.writeShellScriptBin "vllm-b200" ''
    #!/usr/bin/env bash
    set -euo pipefail

    # B200-optimized vLLM configuration
    # These settings maximize throughput and context length for B200

    export CUDA_VISIBLE_DEVICES=0
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1

    # Optimize for B200 architecture
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVIDIA_TF32_OVERRIDE=0

    # Calculate optimal batch sizes for B200
    TOTAL_VRAM_MB=$(( ${toString b200Specs.totalVramGB} * 1024 ))
    MODEL_VRAM_MB=$(( TOTAL_VRAM_MB * ${toString (builtins.floor (gpuMemoryUtilization * 100))} / 100 ))

    # Max sequences based on VRAM and performance
    if [[ -z "''${MAX_NUM_SEQS:-}" ]]; then
      MAX_NUM_SEQS=256
    fi

    if [[ -z "''${MAX_MODEL_LEN:-}" ]]; then
      MAX_MODEL_LEN=${toString maxModelLength}
    fi

    if [[ -z "''${GPU_MEMORY_UTILIZATION:-}" ]]; then
      GPU_MEMORY_UTILIZATION=${toString gpuMemoryUtilization}
    fi

    echo "🚀 Starting vLLM with B200 optimizations"
    echo "========================================"
    echo "Total VRAM: ${toString b200Specs.totalVramGB}GB"
    echo "Tensor Parallel Size: ${toString optimalTensorParallelSize}"
    echo "Pipeline Parallel Size: ${toString optimalPipelineParallelSize}"
    echo "Max Model Length: $MAX_MODEL_LEN tokens"
    echo "Max Sequences: $MAX_NUM_SEQS"
    echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
    echo ""

    exec ${pkgs.vllm}/bin/python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size ${toString optimalTensorParallelSize} \
      --pipeline-parallel-size ${toString optimalPipelineParallelSize} \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --enforce-eager \
      --disable-custom-all-reduce \
      --max-parallel-loading-workers 2 \
      --dtype float16 \
      --quantization fp8 \
      --enable-prefix-caching \
      --block-size 16 \
      --swap-space 64 \
      "$@"
  '';

  # B200-optimized llama.cpp service
  llamacppService = pkgs.writeShellScriptBin "llamacpp-b200" ''
    #!/usr/bin/env bash
    set -euo pipefail

    # B200-optimized llama.cpp configuration
    # Load all layers on GPU for maximum performance

    export CUDA_VISIBLE_DEVICES=0
    export GGML_CUDA_MAX_STREAMS=32

    # B200-specific CUDA optimizations
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export GGML_CUDA_MMQ=1  # Use matrix multiplication kernels
    export GGML_CUDA_F16=1  # Use FP16 acceleration

    echo "🚀 Starting llama.cpp with B200 optimizations"
    echo "============================================"
    echo "GPU Layers: 999 (all layers on GPU)"
    echo "Context Length: 32768 tokens"
    echo "Batch Size: 2048"
    echo "Threads: Auto-detected"
    echo "Flash Attention: Enabled"
    echo "CUDA Graphs: Enabled"
    echo ""

    # Calculate optimal thread count
    CPU_CORES=$(nproc)
    BATCH_THREADS=$(( CPU_CORES > 16 ? 16 : CPU_CORES ))
    MAIN_THREADS=$(( CPU_CORES - BATCH_THREADS ))

    exec ${pkgs.llama-cpp}/bin/llama-server \
      --host 0.0.0.0 \
      --port 8080 \
      --model "$1" \
      --n-gpu-layers 999 \
      --ctx-size 32768 \
      --batch-size 2048 \
      --ubatch-size 512 \
      --parallel 8 \
      --n-predict -1 \
      --cont-batching \
      --flash-attn \
      --no-mmap \
      --mlock \
      --numa \
      --tensor-split 0,0,0,0 \
      --main-gpu 0 \
      --threads "$MAIN_THREADS" \
      --threads-batch "$BATCH_THREADS" \
      --grp-attn-n 8 \
      --grp-attn-w 1 \
      --rope-freq-base 10000 \
      --rope-freq-scale 1 \
      --verbose
  '';

  # Performance monitoring script for B200
  b200Monitor = pkgs.writeShellScriptBin "tf-b200-monitor" ''
    #!/usr/bin/env bash
    set -euo pipefail

    echo "📊 TensorForge B200 Performance Monitor"
    echo "======================================="
    echo "Timestamp: $(date -Iseconds)"
    echo ""

    # GPU metrics
    echo "🎮 GPU Metrics (NVIDIA B200):"
    if command -v nvidia-smi &> /dev/null; then
      nvidia-smi \
        --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit \
        --format=csv,noheader \
        | while IFS=, read -r name total used free gpu_util mem_util temp power_draw power_limit; do
          echo "  GPU: $name"
          echo "    VRAM: $used / $total ($((used * 100 / total))% used)"
          echo "    Utilization: GPU: $gpu_util%, Memory: $mem_util%"
          echo "    Temperature: $temp°C"
          echo "    Power: $power_draw / $power_limit W"
        done
    else
      echo "  nvidia-smi not available"
    fi

    echo ""

    # System metrics
    echo "🖥️  System Metrics:"
    echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 "%"}')"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2 " (" $3/$2*100 "%)"}')"
    echo "  Load: $(uptime | awk -F'load average:' '{print $2}')"

    echo ""

    # TensorForge service status
    echo "🔧 TensorForge Services:"
    if systemctl is-active --quiet vllm-b200; then
      echo "  vLLM: ✅ Running"
    else
      echo "  vLLM: ❌ Stopped"
    fi

    if systemctl is-active --quiet llamacpp-b200; then
      echo "  llama.cpp: ✅ Running"
    else
      echo "  llama.cpp: ❌ Stopped"
    fi

    if systemctl is-active --quiet tensorforge; then
      echo "  TensorForge API: ✅ Running"
      echo "    Endpoint: http://localhost:8080"
      echo "    Metrics: http://localhost:9091"
    else
      echo "  TensorForge API: ❌ Stopped"
    fi

    echo ""
    echo "📈 Performance Recommendations:"
    echo "  1. For 70B+ models: Use --tensor-parallel-size 4"
    echo "  2. For long context: Set --max-model-len 131072"
    echo "  3. For batch processing: Use --max-num-seqs 256"
    echo "  4. Enable FP8 quantization for 2x throughput"
  '';

in
{
  options.services.tensorforge.b200Optimized = {
    enable = mkEnableOption "B200-optimized TensorForge configuration";

    vllm = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable B200-optimized vLLM service";
      };

      modelDir = mkOption {
        type = types.path;
        default = "/var/lib/tensorforge/models";
        description = "Directory containing vLLM models";
      };

      extraArgs = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Extra arguments for vLLM server";
      };
    };

    llamacpp = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable B200-optimized llama.cpp service";
      };

      modelDir = mkOption {
        type = types.path;
        default = "/var/lib/tensorforge/models/gguf";
        description = "Directory containing GGUF models";
      };

      extraArgs = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Extra arguments for llama.cpp server";
      };
    };

    systemTuning = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable B200-specific system performance tuning";
      };

      hugePages = mkOption {
        type = types.bool;
        default = true;
        description = "Enable huge pages for better memory performance";
      };

      cpuGovernor = mkOption {
        type = types.enum [
          "performance"
          "powersave"
          "ondemand"
        ];
        default = "performance";
        description = "CPU frequency governor";
      };

      ioPriority = mkOption {
        type = types.enum [
          "realtime"
          "high"
          "normal"
          "low"
        ];
        default = "high";
        description = "I/O priority for TensorForge processes";
      };
    };

    monitoring = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable B200-specific monitoring";
      };

      prometheusPort = mkOption {
        type = types.port;
        default = 9091;
        description = "Prometheus metrics port";
      };

      grafanaDashboard = mkOption {
        type = types.bool;
        default = true;
        description = "Install B200-optimized Grafana dashboard";
      };
    };
  };

  config = mkIf cfg.enable {
    # System performance tuning for B200
    boot.kernel.sysctl = mkIf cfg.systemTuning.enable {
      # Network tuning for high-throughput
      "net.core.rmem_max" = 134217728;
      "net.core.wmem_max" = 134217728;
      "net.ipv4.tcp_rmem" = "4096 87380 134217728";
      "net.ipv4.tcp_wmem" = "4096 65536 134217728";

      # VM tuning for large memory systems
      "vm.swappiness" = 10;
      "vm.dirty_ratio" = 10;
      "vm.dirty_background_ratio" = 5;
      "vm.overcommit_memory" = 1;
      "vm.overcommit_ratio" = 100;

      # File system tuning
      "fs.file-max" = 2097152;
      "fs.nr_open" = 2097152;
    };

    # Huge pages for better memory performance
    boot.kernelParams = mkIf (cfg.systemTuning.enable && cfg.systemTuning.hugePages) [
      "hugepagesz=1GB"
      "hugepages=64"
      "transparent_hugepage=always"
    ];

    # CPU performance governor
    powerManagement.cpuFreqGovernor = mkIf cfg.systemTuning.enable cfg.systemTuning.cpuGovernor;

    # B200-optimized vLLM service
    systemd.services.vllm-b200 = mkIf cfg.vllm.enable {
      description = "B200-optimized vLLM inference server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      serviceConfig = {
        Type = "simple";
        User = "tensorforge";
        Group = "tensorforge";
        ExecStart = "${vllmService}/bin/vllm-b200 ${escapeShellArgs cfg.vllm.extraArgs}";
        Restart = "always";
        RestartSec = "10s";
        Environment = [
          "CUDA_VISIBLE_DEVICES=0"
          "NCCL_DEBUG=WARN"
          "TORCH_CUDNN_V8_API_ENABLED=1"
        ];

        # Resource limits optimized for B200
        LimitNOFILE = 1048576;
        LimitNPROC = 65536;
        LimitMEMLOCK = "unlimited";

        # I/O priority
        IOSchedulingClass = cfg.systemTuning.ioPriority;

        # Security hardening
        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [
          cfg.vllm.modelDir
          "/tmp"
        ];

        # GPU access
        DeviceAllow = [
          "/dev/nvidia0 rw"
          "/dev/nvidiactl rw"
          "/dev/nvidia-uvm rw"
          "/dev/nvidia-modeset rw"
        ];
        SupplementaryGroups = [
          "nvidia"
          "video"
        ];
      };

      environment = {
        HF_HOME = "/var/lib/tensorforge/cache";
        TORCH_HOME = "/var/lib/tensorforge/torch";
        PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512";
      };

      preStart = ''
        mkdir -p ${cfg.vllm.modelDir}
        mkdir -p /var/lib/tensorforge/{cache,torch}
        chown -R tensorforge:tensorforge /var/lib/tensorforge
      '';
    };

    # B200-optimized llama.cpp service
    systemd.services.llamacpp-b200 = mkIf cfg.llamacpp.enable {
      description = "B200-optimized llama.cpp inference server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      serviceConfig = {
        Type = "simple";
        User = "tensorforge";
        Group = "tensorforge";
        ExecStart = "${llamacppService}/bin/llamacpp-b200 ${escapeShellArgs cfg.llamacpp.extraArgs}";
        Restart = "always";
        RestartSec = "10s";
        WorkingDirectory = cfg.llamacpp.modelDir;

        # Resource limits
        LimitNOFILE = 1048576;
        LimitNPROC = 65536;
        LimitMEMLOCK = "unlimited";

        # I/O priority
        IOSchedulingClass = cfg.systemTuning.ioPriority;

        # Security hardening
        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;

        # GPU access
        DeviceAllow = [
          "/dev/nvidia0 rw"
          "/dev/nvidiactl rw"
          "/dev/nvidia-uvm rw"
          "/dev/nvidia-modeset rw"
        ];
        SupplementaryGroups = [
          "nvidia"
          "video"
        ];
      };

      preStart = ''
        mkdir -p ${cfg.llamacpp.modelDir}
        chown -R tensorforge:tensorforge ${cfg.llamacpp.modelDir}
      '';
    };

    # B200 monitoring service
    systemd.services.tf-b200-monitor = mkIf cfg.monitoring.enable {
      description = "TensorForge B200 performance monitor";
      serviceConfig = {
        Type = "oneshot";
        User = "tensorforge";
        Group = "tensorforge";
        ExecStart = "${b200Monitor}/bin/tf-b200-monitor";
      };
    };

    systemd.timers.tf-b200-monitor = mkIf cfg.monitoring.enable {
      description = "Timer for B200 performance monitoring";
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "*:0/5"; # Every 5 minutes
        Persistent = true;
      };
    };

    # Prometheus exporter for B200 metrics
    services.prometheus.exporters.node = mkIf cfg.monitoring.enable {
      enable = true;
      port = cfg.monitoring.prometheusPort;
      enabledCollectors = [
        "cpu"
        "meminfo"
        "diskstats"
        "filesystem"
        "netdev"
        "stat"
        "systemd"
        "textfile"
      ];
      extraFlags = [
        "--collector.textfile.directory=/var/lib/prometheus/node-exporter"
        "--collector.gpu"
        "--collector.nvidia"
      ];
    };

    # Grafana dashboard for B200
    services.grafana = mkIf (cfg.monitoring.enable && cfg.monitoring.grafanaDashboard) {
      enable = true;
      settings = {
        server = {
          http_port = 3000;
          domain = "localhost";
        };
      };
      provision = {
        enable = true;
        datasources = [
          {
            name = "Prometheus";
            type = "prometheus";
            url = "http://localhost:${toString cfg.monitoring.prometheusPort}";
            isDefault = true;
          }
        ];
        dashboards = [
          {
            name = "TensorForge B200";
            options = {
              path = ./b200-dashboard.json;
            };
          }
        ];
      };
    };

    # User and group for TensorForge services
    users.users.tensorforge = {
      isSystemUser = true;
      group = "tensorforge";
      description = "TensorForge service user";
      home = "/var/lib/tensorforge";
      createHome = true;
      extraGroups = [
        "nvidia"
        "video"
      ];
    };

    users.groups.tensorforge = { };

    # Create necessary directories
    systemd.tmpfiles.rules = [
      "d /var/lib/tensorforge 0755 tensorforge tensorforge -"
      "d /var/lib/tensorforge/models 0755 tensorforge tensorforge -"
      "d /var/lib/tensorforge/models/gguf 0755 tensorforge tensorforge -"
      "d /var/lib/tensorforge/cache 0755 tensorforge tensorforge -"
      "d /var/lib/tensorforge/torch 0755 tensorforge tensorforge -"
      "d /var/lib/tensorforge/logs 0755 tensorforge tensorforge -"
      "d /var/lib/prometheus/node-exporter 0755 prometheus prometheus -"
    ];

    # Install monitoring tools
    environment.systemPackages =
      with pkgs;
      [
        b200Monitor
        nvtop
        nvidia-smi
        dstat
        iotop
        htop
      ]
      ++ (
        if cfg.monitoring.enable then
          [
            prometheus
            grafana
          ]
        else
          [ ]
      );

    # Shell aliases for B200 management
    programs.bash.shellAliases = {
      tf-b200-status = "sudo systemctl status vllm-b200 llamacpp-b200";
      tf-b200-logs = "sudo journalctl -u vllm-b200 -u llamacpp-b200 -n 100 -f";
      tf-b200-restart = "sudo systemctl restart vllm-b200 llamacpp-b200";
      tf-b200-monitor = "${b200Monitor}/bin/tf-b200-monitor";
      tf-b200-gpu = "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv";
    };

    # Documentation
    documentation.nixos.includeAllModules = true;
    documentation.nixos.options = {
      # Add B200-specific options documentation
      "services.tensorforge.b200Optimized" = {
        description = ''
          B200-optimized TensorForge configuration for NVIDIA B200 GPUs.

          This module provides hardware-specific optimizations for running
          TensorForge on NVIDIA B200 GPUs with 192GB HBM3e memory.

          Key optimizations include:
          1. Tensor parallelism optimized for B200's 4-die architecture
          2. vLLM configuration for 131k token context length
          3. llama.cpp configuration with all layers on GPU
          4. System tuning for high-throughput inference pipelines
          5. B200-specific monitoring and metrics collection

          Example configuration:
          ```nix
          services.tensorforge.b200Optimized = {
            enable = true;
            vllm.enable = true;
            llamacpp.enable = true;
            systemTuning.enable = true;
            monitoring.enable = true;
          };
          ```
        '';
      };
    };

    # Hardware optimizations
    nixpkgs.config = {
      cudaSupport = true;
      cudaCapabilities = [ "8.9" ]; # B200 compute capability
      cudaForwardCompat = true;
    };

    hardware.opengl = {
      enable = true;
      driSupport = true;
      driSupport32Bit = true;
      extraPackages = with pkgs; [
        vaapiVdpau
        libvdpau-va-gl
      ];
    };

    # NVIDIA driver configuration
    services.xserver.videoDrivers = [ "nvidia" ];
    hardware.nvidia = {
      modesetting.enable = true;
      powerManagement.enable = true;
      powerManagement.finegrained = true;
      open = false;
      nvidiaSettings = true;
      package = config.boot.kernelPackages.nvidiaPackages.stable;

      # B200-specific driver settings
      prime = {
        offload = {
          enable = true;
          enableOffloadCmd = true;
        };
      };
    };
  };
}
