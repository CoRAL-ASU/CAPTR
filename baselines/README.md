# Refactored Baselines
Implemented baselines:
- Interleaved Reasoner Baseline
- Partial Input Baseline
- Retrieval **TODO: explain**

The interleaved reasoner and partial inpput baseline are refactored version of the baselines used in the MMTabQA paper. We refactored them to use our Hugging Face dataset loading function instead of manually loading data files. Big Changes:
1. Biggest change: **Replaced manual data loading** with the `load_mmtabqa_dataset` function from `load_mmtabqa_utils.py`
2. **Simplified image handling** by leveraging the partial input baseline functionality in the HF dataset loader
3. **vLLM:** Now using vLLM models
4. **Evalution:** automatically run evaluation in the file. Now we don't need multiple step-files anymore like it is in the original MMTabQA code.#

**Important:** The original interleaved baseline uses gpt-4o-mini and provides the opneai api the information that it should use low image resolution. This means that each image will be a maximum of 512x512 pixels and encoded into 85 tokens.



```bash
cd mmtabqa/baselines

# Interleaved baseline (no pruning)
python interleaved_baseline.py --model="google/gemma-3-27b-it" --num_gpus=2

# Partial input baseline
python partial_input_baseline.py --model="google/gemma-3-27b-it" --num_gpus=2

# Retrieval baseline
python retrieval.py --model="google/gemma-3-27b-it" --num_gpus=2
```