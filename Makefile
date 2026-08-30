.PHONY: check format help

check_dirs := datasets generation nsql prompts scripts utils mmtabqa statistics sparsevlm

check:
	ruff check $(check_dirs)  # for lint
	ruff format --check $(check_dirs)  # for checking format

format:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

help:
	@echo "make check: check the code"
	@echo "make format: format the code"
