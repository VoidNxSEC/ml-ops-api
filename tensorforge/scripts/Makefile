# =============================================================================
# tensorforge/scripts/Makefile
# Shortcuts for the full inference pipeline
# =============================================================================

PREFIX      ?= /opt/tensorforge/llamacpp
MODELS_DIR  ?= /var/lib/tensorforge/models/gguf
MODEL       ?= qwen2.5-coder-32b
PORT        ?= 8080
PROMPT      ?= "Hello, what can you do?"
OUTPUT      ?= output.json
FORMAT      ?= text

.PHONY: help bootstrap pull-model start stop restart status logs gpu infer run run-batch health export clean

help:
	@echo "tensorforge inference pipeline"
	@echo ""
	@echo "  Setup:"
	@echo "    make bootstrap              Build llama.cpp with CUDA"
	@echo "    make pull-model MODEL=...   Download GGUF model"
	@echo "    make pull-model MODEL=list  List available models"
	@echo ""
	@echo "  Server:"
	@echo "    make start                  Start server (auto-detect model)"
	@echo "    make stop                   Stop server"
	@echo "    make restart                Restart server"
	@echo "    make status                 Show server + GPU status"
	@echo "    make logs                   Tail server logs"
	@echo "    make gpu                    Show GPU stats"
	@echo ""
	@echo "  Inference:"
	@echo "    make infer PROMPT='...'     Single inference (server must be running)"
	@echo "    make run PROMPT='...'       Full pipeline: start → infer → stop"
	@echo "    make run-batch BATCH=dir/   Batch all .txt files in dir"
	@echo ""
	@echo "  Results:"
	@echo "    make health                 Comprehensive health check"
	@echo "    make export FORMAT=markdown Convert output.json to markdown"
	@echo ""
	@echo "  Variables:"
	@echo "    MODEL=$(MODEL)"
	@echo "    PORT=$(PORT)"
	@echo "    OUTPUT=$(OUTPUT)"

bootstrap:
	PREFIX=$(PREFIX) MODELS_DIR=$(MODELS_DIR) bash $(CURDIR)/bootstrap.sh

pull-model:
	@if [ "$(MODEL)" = "list" ]; then \
		TF_MODELS=$(MODELS_DIR) bash $(CURDIR)/model-pull.sh --list; \
	else \
		TF_MODELS=$(MODELS_DIR) bash $(CURDIR)/model-pull.sh --model $(MODEL); \
	fi

start:
	TF_PORT=$(PORT) bash $(CURDIR)/server.sh start

stop:
	bash $(CURDIR)/server.sh stop

restart:
	TF_PORT=$(PORT) bash $(CURDIR)/server.sh restart

status:
	TF_PORT=$(PORT) bash $(CURDIR)/server.sh status

logs:
	bash $(CURDIR)/server.sh logs --follow

gpu:
	bash $(CURDIR)/server.sh gpu

infer:
	TF_PORT=$(PORT) bash $(CURDIR)/infer.sh --prompt $(PROMPT) --output $(OUTPUT)
	@echo ""
	@cat $(OUTPUT) | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"

run:
	TF_PORT=$(PORT) bash $(CURDIR)/run.sh \
		--prompt $(PROMPT) \
		--output $(OUTPUT)

run-batch:
	TF_PORT=$(PORT) bash $(CURDIR)/run.sh \
		--batch $(BATCH) \
		--output-dir $(OUTPUT_DIR)

health:
	TF_PORT=$(PORT) bash $(CURDIR)/health.sh

health-json:
	TF_PORT=$(PORT) bash $(CURDIR)/health.sh --json | jq .

health-watch:
	TF_PORT=$(PORT) bash $(CURDIR)/health.sh --watch 5

export:
	bash $(CURDIR)/export.sh --input $(OUTPUT) --format $(FORMAT)

clean:
	@echo "Remove logs and temp files?"
	@read -p "Confirm [y/N]: " c && [ "$$c" = "y" ] && \
		rm -f $(PREFIX)/logs/*.log $(PREFIX)/run/*.pid || true
