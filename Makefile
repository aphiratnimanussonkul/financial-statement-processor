.PHONY: help install install-uv install-pip setup run clean test format lint

# Variables
PYTHON := python3
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python
PIP_VENV := $(VENV)/bin/pip
UV := uv
SCRIPT := index.py

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## แสดงคำแนะนำการใช้งาน
	@echo "$(GREEN)Financial Statement Processor - Makefile Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install          # ติดตั้ง dependencies"
	@echo "  make run PDF=file.pdf # ประมวลผล PDF file"
	@echo "  make run PDF=file.pdf OUTPUT=output.json # ระบุ output file"

setup: ## สร้าง virtual environment (ถ้ายังไม่มี)
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
		echo "$(GREEN)Virtual environment created!$(NC)"; \
	else \
		echo "$(GREEN)Virtual environment already exists$(NC)"; \
	fi

install-uv: setup ## ติดตั้ง dependencies ด้วย uv (แนะนำ)
	@if command -v $(UV) > /dev/null 2>&1; then \
		echo "$(YELLOW)Installing dependencies with uv...$(NC)"; \
		$(UV) sync; \
		echo "$(GREEN)Dependencies installed successfully!$(NC)"; \
	else \
		echo "$(RED)Error: uv is not installed$(NC)"; \
		echo "$(YELLOW)Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh$(NC)"; \
		echo "$(YELLOW)Or use: make install-pip$(NC)"; \
		exit 1; \
	fi

install-pip: setup ## ติดตั้ง dependencies ด้วย pip
	@echo "$(YELLOW)Installing dependencies with pip...$(NC)"; \
	$(PIP_VENV) install --upgrade pip; \
	$(PIP_VENV) install -r requirements.txt 2>/dev/null || $(PIP_VENV) install pdfplumber anthropic python-dotenv pandas pydantic; \
	echo "$(GREEN)Dependencies installed successfully!$(NC)"

install: install-uv ## ติดตั้ง dependencies (ใช้ uv ถ้ามี, ไม่งั้นใช้ pip)

run: ## ประมวลผล PDF file (ใช้: make run PDF=file.pdf [OUTPUT=output.json])
	@if [ -z "$(PDF)" ]; then \
		echo "$(RED)Error: Please specify PDF file$(NC)"; \
		echo "$(YELLOW)Usage: make run PDF=file.pdf [OUTPUT=output.json]$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(PDF)" ]; then \
		echo "$(RED)Error: PDF file not found: $(PDF)$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make install' first$(NC)"; \
		exit 1; \
	fi
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "$(YELLOW)Warning: ANTHROPIC_API_KEY not set in environment$(NC)"; \
		echo "$(YELLOW)The script will try to load from .env file$(NC)"; \
	fi
	@echo "$(YELLOW)Processing $(PDF)...$(NC)"
	@if [ -n "$(OUTPUT)" ]; then \
		$(PYTHON_VENV) $(SCRIPT) "$(PDF)" --output "$(OUTPUT)"; \
	else \
		$(PYTHON_VENV) $(SCRIPT) "$(PDF)"; \
	fi

run-verbose: ## ประมวลผล PDF file แบบ verbose (ใช้: make run-verbose PDF=file.pdf)
	@if [ -z "$(PDF)" ]; then \
		echo "$(RED)Error: Please specify PDF file$(NC)"; \
		echo "$(YELLOW)Usage: make run-verbose PDF=file.pdf$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(PDF)" ]; then \
		echo "$(RED)Error: PDF file not found: $(PDF)$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make install' first$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Processing $(PDF) (verbose mode)...$(NC)"
	@if [ -n "$(OUTPUT)" ]; then \
		$(PYTHON_VENV) $(SCRIPT) "$(PDF)" --output "$(OUTPUT)" --verbose; \
	else \
		$(PYTHON_VENV) $(SCRIPT) "$(PDF)" --verbose; \
	fi

clean: ## ลบ cache files และ __pycache__
	@echo "$(YELLOW)Cleaning cache files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@echo "$(GREEN)Clean completed!$(NC)"

clean-all: clean ## ลบ virtual environment และ cache files ทั้งหมด
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	@rm -rf $(VENV)
	@echo "$(GREEN)All cleaned!$(NC)"

test: ## ทดสอบการทำงาน (ตัวอย่าง)
	@echo "$(YELLOW)Testing installation...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make install' first$(NC)"; \
		exit 1; \
	fi
	@$(PYTHON_VENV) -c "import pdfplumber, anthropic, pydantic; print('$(GREEN)All dependencies are installed correctly!$(NC)')"

format: ## Format code ด้วย black (ถ้ามี)
	@if command -v black > /dev/null 2>&1; then \
		black $(SCRIPT); \
		echo "$(GREEN)Code formatted!$(NC)"; \
	else \
		echo "$(YELLOW)black is not installed. Install with: pip install black$(NC)"; \
	fi

lint: ## ตรวจสอบ code ด้วย pylint (ถ้ามี)
	@if command -v pylint > /dev/null 2>&1; then \
		pylint $(SCRIPT); \
	else \
		echo "$(YELLOW)pylint is not installed. Install with: pip install pylint$(NC)"; \
	fi

.DEFAULT_GOAL := help

