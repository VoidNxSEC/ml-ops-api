# =============================================================================
# ml-ops-api/arch-analyzer/Makefile
# =============================================================================

PLATFORM_ROOT ?= $(HOME)/master
OUTPUT_DIR    ?= $(PLATFORM_ROOT)/arch-analyzer
MODEL         ?= qwen2.5-coder-32b
TARGET        ?= $(PLATFORM_ROOT)/neoland
TEMPLATE      ?= auto

.PHONY: help platform analyze-project list-templates setup clean

help:
	@echo "arch-analyzer — AI-powered code analysis"
	@echo ""
	@echo "  make platform                    Analyze entire ~/master/ platform"
	@echo "  make platform MODEL=llama-3.3-70b"
	@echo ""
	@echo "  make analyze-project TARGET=path  Analyze single project"
	@echo "  make analyze-project TARGET=~/master/neoland TEMPLATE=rust"
	@echo ""
	@echo "  make list-templates              Show available language templates"
	@echo "  make setup                       Install Python deps"
	@echo ""
	@echo "  Variables:"
	@echo "    PLATFORM_ROOT = $(PLATFORM_ROOT)"
	@echo "    OUTPUT_DIR    = $(OUTPUT_DIR)"
	@echo "    MODEL         = $(MODEL)"
	@echo "    LLAMACPP_URL  = $${LLAMACPP_URL:-http://127.0.0.1:8080}"

platform:
	@mkdir -p $(OUTPUT_DIR)
	LLM_MODEL=$(MODEL) bash $(CURDIR)/analyze.sh platform \
		--path $(PLATFORM_ROOT) \
		--output $(OUTPUT_DIR) \
		--model $(MODEL)

analyze-project:
	@mkdir -p $(OUTPUT_DIR)
	LLM_MODEL=$(MODEL) bash $(CURDIR)/analyze.sh analyze \
		--path $(TARGET) \
		--output $(OUTPUT_DIR) \
		--template $(TEMPLATE) \
		--model $(MODEL)

list-templates:
	bash $(CURDIR)/analyze.sh list-templates

setup:
	python3 -m venv $(CURDIR)/.venv
	$(CURDIR)/.venv/bin/pip install -q \
		aiohttp aiofiles pydantic rich typer httpx pyyaml psutil

clean:
	rm -rf $(CURDIR)/.venv
	rm -f $(OUTPUT_DIR)/PLATFORM-REPORT*.md $(OUTPUT_DIR)/PLATFORM-REPORT*.json
