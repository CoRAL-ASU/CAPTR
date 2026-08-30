#!/bin/bash

set -euo pipefail

if [ -f .env ]; then
	set -a
	. ./.env
	set +a
fi

run_config_group() {
	local group_name="$1"
	shift

	local index=1
	local total=$#
	for config_file in "$@"; do
		echo "[$index/$total] Running ${group_name} strategy: ${config_file}"
		python3 run_model.py --config_file="${config_file}"
		index=$((index + 1))
	done
}

run_config_group "Gemma3" \
	"run_configs/gemma3/detailed_captioned_interleaved_reasoner.json" \
	"run_configs/gemma3/naive_captioned_interleaved_reasoner.json" \
	"run_configs/gemma3/short_captioned_interleaved_reasoner.json"

run_config_group "Qwen3-VL" \
	"run_configs/qwen3/detailed_captioned_interleaved_reasoner.json" \
	"run_configs/qwen3/naive_captioned_interleaved_reasoner.json" \
	"run_configs/qwen3/short_captioned_interleaved_reasoner.json"

run_config_group "Qwen3.5" \
	"run_configs/qwen35/detailed_captioned_interleaved_reasoner.json" \
	"run_configs/qwen35/naive_captioned_interleaved_reasoner.json" \
	"run_configs/qwen35/short_captioned_interleaved_reasoner.json"
