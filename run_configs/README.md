# Configuration

Experiments are controlled via JSON configuration files in `run_configs/`. Key parameters:

```json
{
    "engine": "vllm",                    // Inference engine (vllm, gpt)
    "vllm_model_name": "google/gemma-3-27b-it",
    "dataset": "all_multimodal_datasets", // or specific dataset
    "dataset_version": "captioned",       // use captioned version
    "caption_type": "short_with_context", // naive, short_with_context, detailed
    "load_images_for_reasoning": true,    // Load images for final reasoning
    "run_steps": ["step1", "step2", "step3", "step4", "step6"],
    "skip_steps": ["step5"],              // Skip SQL reasoning step
    "run_final_evaluation": true
}