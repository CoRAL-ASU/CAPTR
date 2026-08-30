# Generation Module
This directory provides wrappers to call LLMs (via API or vllm) to complete (batches) of prompts.

## Core Components

### 1. `generator.py` - Abstract Base Class
Defines the common interface that all generators must implement:
- `prompt_row_truncate()` - Fit prompts within token limits by truncating table rows
- `build_few_shot_prompt_from_file()` - Build few-shot prompts from template files
- `build_generate_prompt()` - Build generation prompts for specific tasks
- `generate_one_pass()` - Generate single responses
- `generate_batch_pass()` - Generate responses in batches

### 2. `prompt.py` - Prompt Builder
Handles the construction of prompts for table-based reasoning tasks:
- **Table Formatting**: Multiple styles (`text_full_table`, `create_table_select_3`, `transpose`)
- **Prompt Types**: Support for different generation tasks (answer, column selection, row selection, verification)
- **Few-shot Learning**: Build demonstration examples from tables and questions

### 3. Model-Specific Generators
- generator_gpt.py for OpenAI API. We used this in the beginning but then pivoted to vllm.
- generator_vllm_qwen.py for vllm (but since their vision models don't perform well, we chose to use gemma3)
- generator_vllm_gemma3.py for gemma3