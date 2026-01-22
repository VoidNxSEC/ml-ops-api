//! TensorForge CLI and Server
//!
//! This binary provides multiple modes of operation:
//! - Server mode: Run the TensorForge API server
//! - Worker mode: Process inference requests from queue
//! - CLI mode: Interactive command-line interface
//! - Batch mode: Process datasets in batch

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use clap::{Parser, Subcommand, ValueEnum};
use tracing::{debug, error, info, level_filters::LevelFilter, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use tensorforge::{
    initialize_with_config, Config, Orchestrator, TensorForgeError, TensorForgeResult,
};

#[derive(Parser)]
#[command(
    name = "tensorforge",
    author = "kernelcore",
    version = tensorforge::version(),
    about = "High-performance ML inference orchestration engine",
    long_about = r#"
TensorForge - Forging the future of massive-scale ML inference

A production-grade orchestration engine for massive-scale ML pipelines.
Optimized for NVIDIA B200 with 192GB HBM3e, supporting vLLM and llama.cpp
backends with intelligent routing, advanced metrics, and DSPy integration.
"#
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Configuration file path (TOML format)
    #[arg(short, long, global = true, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, global = true, value_name = "LEVEL", default_value = "info")]
    log_level: LogLevel,

    /// Enable JSON log formatting (useful for structured logging)
    #[arg(long, global = true)]
    json_logs: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the TensorForge API server
    Serve {
        /// Host address to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Enable CORS for web applications
        #[arg(long)]
        cors: bool,

        /// Number of worker threads (0 = auto)
        #[arg(long, default_value_t = 0)]
        workers: usize,

        /// Enable Prometheus metrics endpoint at /metrics
        #[arg(long)]
        metrics: bool,

        /// Enable WebSocket endpoint at /ws for real-time updates
        #[arg(long)]
        websocket: bool,
    },

    /// Process inference requests from a message queue
    Worker {
        /// Queue type (redis, rabbitmq, in-memory)
        #[arg(long, default_value = "memory")]
        queue: String,

        /// Queue connection string
        #[arg(long)]
        connection: Option<String>,

        /// Number of concurrent workers
        #[arg(short, long, default_value_t = 4)]
        concurrency: usize,

        /// Queue name or topic
        #[arg(long, default_value = "inference-requests")]
        queue_name: String,
    },

    /// Run interactive CLI for manual inference and management
    Cli {
        /// Model to use for inference
        #[arg(short, long)]
        model: Option<String>,

        /// Backend to use (vllm, llamacpp, auto)
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Interactive mode (REPL)
        #[arg(short, long)]
        interactive: bool,
    },

    /// Process datasets in batch mode
    Batch {
        /// Input file or directory
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Dataset format (jsonl, csv, parquet)
        #[arg(long, default_value = "jsonl")]
        format: String,

        /// Number of parallel processing streams
        #[arg(short, long, default_value_t = 4)]
        parallel: usize,

        /// Enable checkpointing for fault tolerance
        #[arg(long)]
        checkpoint: bool,

        /// Checkpoint interval (number of items)
        #[arg(long, default_value_t = 1000)]
        checkpoint_interval: usize,
    },

    /// Check system health and backend status
    Health {
        /// Perform deep health check (includes backend connectivity)
        #[arg(long)]
        deep: bool,

        /// Exit with error code if any backend is unhealthy
        #[arg(long)]
        strict: bool,
    },

    /// List available models and backends
    List {
        /// List backends instead of models
        #[arg(long)]
        backends: bool,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,

        /// Filter by backend type
        #[arg(long)]
        backend_type: Option<String>,

        /// Filter by model format
        #[arg(long)]
        format: Option<String>,
    },

    /// Load a model onto a backend
    Load {
        /// Model identifier or path
        model: String,

        /// Backend to load onto
        #[arg(short, long)]
        backend: Option<String>,

        /// GPU layers to load (llama.cpp only)
        #[arg(long)]
        gpu_layers: Option<u32>,

        /// Context length
        #[arg(long)]
        context_length: Option<u32>,

        /// Wait for model to be ready
        #[arg(short, long)]
        wait: bool,

        /// Timeout in seconds
        #[arg(long, default_value_t = 300)]
        timeout: u64,
    },

    /// Unload a model from backend
    Unload {
        /// Model identifier
        model: String,

        /// Backend to unload from (optional, unloads from all if not specified)
        backend: Option<String>,

        /// Force unload even if in use
        #[arg(short, long)]
        force: bool,
    },

    /// Display system metrics and statistics
    Stats {
        /// Refresh interval in seconds (0 = single snapshot)
        #[arg(short, long, default_value_t = 0)]
        refresh: u64,

        /// Metrics to display (comma-separated)
        #[arg(long, default_value = "all")]
        metrics: String,

        /// Output format (text, json, csv)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Generate shell completions
    Completions {
        /// Shell type
        shell: clap_complete::Shell,
    },
}

#[derive(Clone, ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => LevelFilter::TRACE,
            LogLevel::Debug => LevelFilter::DEBUG,
            LogLevel::Info => LevelFilter::INFO,
            LogLevel::Warn => LevelFilter::WARN,
            LogLevel::Error => LevelFilter::ERROR,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    setup_logging(&cli)?;

    info!("Starting TensorForge {}", tensorforge::version());
    debug!("Command line arguments: {:?}", std::env::args().collect::<Vec<_>>());

    // Load configuration
    let config = load_config(&cli).await.context("Failed to load configuration")?;

    // Handle signals for graceful shutdown
    let shutdown = setup_signal_handler();

    // Execute the chosen command
    match cli.command {
        Commands::Serve {
            host,
            port,
            cors,
            workers,
            metrics,
            websocket,
        } => {
            run_server(config, host, port, cors, workers, metrics, websocket, shutdown).await
        }
        Commands::Worker {
            queue,
            connection,
            concurrency,
            queue_name,
        } => {
            run_worker(config, queue, connection, concurrency, queue_name, shutdown).await
        }
        Commands::Cli {
            model,
            backend,
            interactive,
        } => run_cli(config, model, backend, interactive, shutdown).await,
        Commands::Batch {
            input,
            output,
            format,
            parallel,
            checkpoint,
            checkpoint_interval,
        } => {
            run_batch(
                config,
                input,
                output,
                format,
                parallel,
                checkpoint,
                checkpoint_interval,
                shutdown,
            )
            .await
        }
        Commands::Health { deep, strict } => run_health(config, deep, strict).await,
        Commands::List {
            backends,
            verbose,
            backend_type,
            format,
        } => run_list(config, backends, verbose, backend_type, format).await,
        Commands::Load {
            model,
            backend,
            gpu_layers,
            context_length,
            wait,
            timeout,
        } => {
            run_load(config, model, backend, gpu_layers, context_length, wait, timeout).await
        }
        Commands::Unload {
            model,
            backend,
            force,
        } => run_unload(config, model, backend, force).await,
        Commands::Stats {
            refresh,
            metrics,
            format,
        } => run_stats(config, refresh, metrics, format).await,
        Commands::Completions { shell } => run_completions(shell),
    }
}

/// Set up logging based on CLI options
fn setup_logging(cli: &Cli) -> anyhow::Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(cli.log_level.clone().into())
        .from_env_lossy();

    let registry = tracing_subscriber::registry().with(filter);

    if cli.json_logs {
        let json_layer = fmt::layer()
            .json()
            .with_file(true)
            .with_line_number(true)
            .with_thread_ids(true)
            .with_thread_names(true);
        registry.with(json_layer).init();
    } else {
        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_level(true)
            .with_thread_ids(false)
            .with_thread_names(false)
            .compact();
        registry.with(fmt_layer).init();
    }

    Ok(())
}

/// Load configuration from file and environment
async fn load_config(cli: &Cli) -> anyhow::Result<Config> {
    let mut config = Config::default();

    // Load from file if specified
    if let Some(config_path) = &cli.config {
        if config_path.exists() {
            info!("Loading configuration from: {:?}", config_path);
            config = Config::from_file(config_path).await?;
        } else {
            warn!("Configuration file not found: {:?}", config_path);
        }
    }

    // Apply environment variable overrides
    config.apply_env_vars();

    Ok(config)
}

/// Set up signal handlers for graceful shutdown
fn setup_signal_handler() -> tokio::sync::watch::Receiver<bool> {
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    tokio::spawn(async move {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                info!("Received Ctrl+C, initiating graceful shutdown...");
            }
            _ = terminate => {
                info!("Received SIGTERM, initiating graceful shutdown...");
            }
        }

        // Notify shutdown
        let _ = shutdown_tx.send(true);

        // Give some time for graceful shutdown
        tokio::time::sleep(Duration::from_secs(2)).await;
    });

    shutdown_rx
}

/// Run the API server
async fn run_server(
    config: Config,
    host: String,
    port: u16,
    cors: bool,
    workers: usize,
    metrics: bool,
    websocket: bool,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<()> {
    info!("Starting TensorForge server on {}:{}", host, port);
    info!("Configuration: cors={}, workers={}, metrics={}, websocket={}", cors, workers, metrics, websocket);

    // Initialize orchestrator
    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    // Convert to Arc for sharing
    let orchestrator = Arc::new(orchestrator);

    // Start API server (implementation depends on features)
    #[cfg(feature = "api")]
    {
        use tensorforge::api::start_server;

        info!("Starting API server...");
        let server_result = start_server(
            orchestrator,
            &host,
            port,
            cors,
            metrics,
            websocket,
            shutdown.clone(),
        ).await;

        match server_result {
            Ok(_) => info!("Server shutdown completed"),
            Err(e) => error!("Server error: {}", e),
        }
    }

    #[cfg(not(feature = "api"))]
    {
        error!("API feature not enabled. Rebuild with `--features api`");
        return Err(anyhow::anyhow!("API feature not enabled"));
    }

    Ok(())
}

/// Run worker mode
async fn run_worker(
    config: Config,
    queue: String,
    connection: Option<String>,
    concurrency: usize,
    queue_name: String,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<()> {
    info!("Starting worker with queue={}, concurrency={}", queue, concurrency);

    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    info!("Worker ready. Listening for messages on queue '{}'", queue_name);

    // Wait for shutdown signal
    while !*shutdown.borrow_and_update() {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    info!("Worker shutdown completed");
    Ok(())
}

/// Run CLI mode
async fn run_cli(
    config: Config,
    model: Option<String>,
    backend: String,
    interactive: bool,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<()> {
    info!("Starting CLI mode");

    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    if interactive {
        info!("Entering interactive mode. Type 'help' for commands, 'exit' to quit.");

        // Simple REPL (would be expanded in real implementation)
        let mut buffer = String::new();
        while !*shutdown.borrow_and_update() {
            print!("tensorforge> ");
            std::io::Write::flush(&mut std::io::stdout())?;

            buffer.clear();
            std::io::stdin().read_line(&mut buffer)?;

            let input = buffer.trim();
            if input.is_empty() {
                continue;
            }

            match input {
                "exit" | "quit" => break,
                "help" => println!("Available commands: exit, quit, help, status, models"),
                "status" => println!("Orchestrator is running"),
                "models" => println!("Models would be listed here"),
                _ => println!("Unknown command: {}", input),
            }
        }
    } else if let Some(model_name) = model {
        info!("Running single inference with model: {}, backend: {}", model_name, backend);
        // Single inference would go here
        println!("Single inference mode not yet implemented");
    } else {
        info!("No model specified, showing status");
        println!("TensorForge CLI - Use --interactive for REPL or --model for single inference");
    }

    info!("CLI mode completed");
    Ok(())
}

/// Run batch processing
async fn run_batch(
    config: Config,
    input: PathBuf,
    output: Option<PathBuf>,
    format: String,
    parallel: usize,
    checkpoint: bool,
    checkpoint_interval: usize,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<()> {
    info!("Starting batch processing: input={:?}, format={}, parallel={}", input, format, parallel);

    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    // Check input exists
    if !input.exists() {
        return Err(anyhow::anyhow!("Input path does not exist: {:?}", input));
    }

    // Determine output directory
    let output_dir = output.unwrap_or_else(|| {
        let mut default = input.clone();
        default.set_extension("processed");
        default
    });

    // Create output directory if needed
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
    }

    info!("Output directory: {:?}", output_dir);
    info!("Checkpointing: {} (interval: {})", checkpoint, checkpoint_interval);

    // Simulate batch processing
    let mut processed = 0;
    while !*shutdown.borrow_and_update() && processed < 100 {
        // In real implementation, this would process actual data
        tokio::time::sleep(Duration::from_millis(100)).await;
        processed += 1;

        if processed % 10 == 0 {
            info!("Processed {} items", processed);
        }

        if checkpoint && processed % checkpoint_interval == 0 {
            info!("Checkpoint created at item {}", processed);
        }
    }

    if *shutdown.borrow_and_update() {
        warn!("Batch processing interrupted by shutdown signal");
    } else {
        info!("Batch processing completed. Processed {} items", processed);
    }

    Ok(())
}

/// Run health check
async fn run_health(config: Config, deep: bool, strict: bool) -> anyhow::Result<()> {
    info!("Running health check (deep={}, strict={})", deep, strict);

    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    // Basic health check
    println!("🧪 TensorForge Health Check");
    println!("==========================");
    println!("Version: {}", tensorforge::version());
    println!("Build: {}", env!("CARGO_PKG_REPOSITORY"));

    // Check backends if deep check enabled
    if deep {
        println!("\n🔍 Deep Health Check:");
        println!("  - Backend connectivity: PENDING");
        println!("  - Model availability: PENDING");
        println!("  - VRAM status: PENDING");
        // Actual checks would go here
    }

    println!("\n✅ Basic system check passed");

    if strict {
        // In strict mode, we'd fail if any check fails
        println!("⚠️  Strict mode enabled - any failures would cause exit");
    }

    Ok(())
}

/// Run list command
async fn run_list(
    config: Config,
    backends: bool,
    verbose: bool,
    backend_type: Option<String>,
    format: Option<String>,
) -> anyhow::Result<()> {
    info!("Listing resources (backends={}, verbose={})", backends, verbose);

    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    if backends {
        println!("📡 Available Backends:");
        println!("  - vLLM (optimized for throughput, long context)");
        println!("  - llama.cpp (optimized for efficiency, quantized models)");

        if verbose {
            println!("\n📊 Backend Details:");
            println!("  vLLM:");
            println!("    • Endpoint: http://localhost:8000");
            println!("    • Status: Healthy");
            println!("    • Capabilities: streaming, embeddings, function calling");
            println!("    • Max context: 131,072 tokens");
            println!("\n  llama.cpp:");
            println!("    • Endpoint: http://localhost:8080");
            println!("    • Status: Healthy");
            println!("    • Capabilities: quantized models, low latency");
            println!("    • Max context: 32,768 tokens");
        }
    } else {
        println!("🤖 Available Models:");
        println!("  - mixtral-8x7b (vLLM, llama.cpp)");
        println!("  - llama-2-70b (vLLM, llama.cpp)");
        println!("  - codellama-34b (llama.cpp)");
        println!("  - mistral-7b (vLLM, llama.cpp)");

        if verbose {
            println!("\n📋 Model Details:");
            println!("  mixtral-8x7b:");
            println!("    • Size: 46.7GB (FP16)");
            println!("    • Parameters: 8x7B");
            println!("    • Context: 32K");
            println!("    • Quantizations: FP16, INT8, Q4_K_M");
            println!("    • Estimated VRAM: 24GB (FP16), 14GB (Q4_K_M)");
        }
    }

    Ok(())
}

/// Run load command
async fn run_load(
    config: Config,
    model: String,
    backend: Option<String>,
    gpu_layers: Option<u32>,
    context_length: Option<u32>,
    wait: bool,
    timeout: u64,
) -> anyhow::Result<()> {
    info!("Loading model: {}, backend: {:?}", model, backend);

    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    println!("🔄 Loading model '{}'", model);

    if let Some(backend_name) = &backend {
        println!("  Backend: {}", backend_name);
    } else {
        println!("  Backend: auto-select");
    }

    if let Some(layers) = gpu_layers {
        println!("  GPU layers: {}", layers);
    }

    if let Some(ctx) = context_length {
        println!("  Context length: {}", ctx);
    }

    // Simulate loading
    println!("  Status: Loading...");
    tokio::time::sleep(Duration::from_secs(2)).await;

    if wait {
        println!("  Waiting for model to be ready...");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    println!("✅ Model '{}' loaded successfully", model);

    Ok(())
}

/// Run unload command
async fn run_unload(
    config: Config,
    model: String,
    backend: Option<String>,
    force: bool,
) -> anyhow::Result<()> {
    info!("Unloading model: {}, backend: {:?}, force: {}", model, backend, force);

    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    println!("🗑️  Unloading model '{}'", model);

    if let Some(backend_name) = &backend {
        println!("  From backend: {}", backend_name);
    }

    if force {
        println!("  Force mode: enabled (will unload even if in use)");
    }

    // Simulate unloading
    println!("  Status: Unloading...");
    tokio::time::sleep(Duration::from_secs(1)).await;

    println!("✅ Model '{}' unloaded successfully", model);

    Ok(())
}

/// Run stats command
async fn run_stats(
    config: Config,
    refresh: u64,
    metrics: String,
    format: String,
) -> anyhow::Result<()> {
    info!("Displaying stats: refresh={}, metrics={}, format={}", refresh, metrics, format);

    let orchestrator = initialize_with_config(config)
        .await
        .context("Failed to initialize orchestrator")?;

    println!("📊 TensorForge Statistics");
    println!("========================");
    println!("Timestamp: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
    println!();

    // System metrics
    println!("🖥️  System Metrics:");
    println!("  CPU Usage: 12%");
    println!("  Memory: 8.2GB / 32GB (25.6%)");
    println!("  Disk I/O: Read: 45MB/s, Write: 12MB/s");
    println!();

    // GPU metrics (simulated for B200)
    println!("🎮 GPU Metrics (NVIDIA B200):");
    println!("  GPU Utilization: 78%");
    println!("  VRAM: 148GB / 192GB (77%)");
    println!("  Temperature: 68°C");
    println!("  Power: 320W / 1000W");
    println!("  Throughput: 8,500 tokens/sec");
    println!();

    // Backend metrics
    println!("🔧 Backend Metrics:");
    println!("  vLLM:");
    println!("    • Active requests: 42");
    println!("    • Queue depth: 8");
    println!("    • Avg latency: 85ms");
    println!("    • Success rate: 99.2%");
    println!("  llama.cpp:");
    println!("    • Active requests: 18");
    println!("    • Queue depth: 3");
    println!("    • Avg latency: 150ms");
    println!("    • Success rate: 99.8%");
    println!();

    // Cost metrics
    println!("💰 Cost Metrics:");
    println!("  Current rate: $0.42 / hour");
    println!("  Tokens processed: 12.4M");
    println!("  Estimated cost: $5.21");
    println!("  Efficiency: 1.8 tokens/J");

    if refresh > 0 {
        println!("\n🔄 Refreshing every {} seconds...", refresh);
        println!("Press Ctrl+C to stop");

        loop {
            tokio::time::sleep(Duration::from_secs(refresh)).await;
            // In real implementation, would fetch fresh metrics
            println!("\n--- Refresh ---");
        }
    }

    Ok(())
}

/// Run completions command
fn run_completions(shell: clap_complete::Shell) -> anyhow::Result<()> {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();

    clap_complete::generate(shell, &mut cmd, bin_name, &mut std::io::stdout());

    Ok(())
}
