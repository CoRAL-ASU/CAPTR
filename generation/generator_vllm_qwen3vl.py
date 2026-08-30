"""
vLLM generation wrapper for Qwen3-VL models.
Based on the Gemma3 implementation but adapted for Qwen3-VL architecture.
"""

import os
import time
import inspect
from argparse import Namespace
from typing import Any, Dict, List

import torch
import vllm
from dotenv import load_dotenv
from PIL import Image
from vllm import EngineArgs, TextPrompt

from generation.generator import Generator
from transformers import AutoProcessor, AutoTokenizer


load_dotenv(".env")


def _safe_from_pretrained(loader_fn, model_path, **kwargs):
    """
    Safely load from a local path or hub repo.
    Falls back to direct local loading if the path is a local snapshot.
    """
    try:
        return loader_fn(model_path, **kwargs)
    except Exception as e:
        # Check if it's a local path that transformers couldn't recognize
        if os.path.isdir(model_path):
            print(f"⚠️ from_pretrained failed with {type(e).__name__}, trying direct local load...")
            # For local snapshots, try with allow_patterns to load only what's needed
            try:
                kwargs_local = kwargs.copy()
                kwargs_local["local_files_only"] = True
                # Try without trust_remote_code initially
                return loader_fn(model_path, **kwargs_local)
            except Exception as e2:
                print(f"⚠️ Still failed: {e2}. This may be a deeper issue.")
                raise e from e2
        else:
            raise


class VLLMGeneratorQwen3VL(Generator):
    """
    vLLM generation wrapper for Qwen3-VL models.
    Supports text-image interleaved input for multimodal reasoning.
    """

    def __init__(
        self,
        args,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.85,
        system_prompt="I will give you some x-y examples followed by a x, you need to give me the y, and no other content.",
    ):
        """
        Args:
            args: ArgumentParser object with the following attributes:
                - engine: must be "vllm"
                - vllm_model_name: name of the Qwen3-VL model
                - number_of_gpus: number of GPUs to use for tensor parallelism
                - max_api_total_tokens: maximum context window size
                - prompt_mode: must be "text-image"
            limit_mm_per_prompt: Maximum number of images per prompt
            gpu_memory_utilization: Fraction of GPU memory to use
            system_prompt: System prompt for the model (can be None to disable)
        """
        print(f"Initializing VLLMGeneratorQwen3VL with args: {args}")
        assert args.engine == "vllm", "Generator only supports vllm engine"
        assert args.vllm_model_name, "vllm_model_name is required when engine is vllm"
        assert args.number_of_gpus, "number_of_gpus is required when engine is vllm"
        self.args = args

        # Model configuration
        self.model_name = args.vllm_model_name
        self.tensor_parallel_size = args.number_of_gpus
        self.max_model_len = args.max_api_total_tokens
        self.prompt_mode = args.prompt_mode if args.prompt_mode else "text-image"
        assert self.prompt_mode == "text-image", "prompt_mode must be 'text-image' for Qwen3-VL"

        print(f"torch.get_num_threads(): {torch.get_num_threads()}")

        # Set important variables
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.system_prompt = system_prompt

        # Initialize model
        self.gpu_memory_utilization = gpu_memory_utilization
        self._load_model()

    def _load_model(self):
        """Load the Qwen3-VL model and processor using vLLM."""
        # Detect if model_name is a local path
        is_local_path = os.path.isdir(self.model_name)
        
        # Load processor and tokenizer with safe fallback
        processor_kwargs = {
            "trust_remote_code": True,
        }
        if not is_local_path:
            processor_kwargs["token"] = os.getenv("HF_TOKEN")
            
        self.processor = _safe_from_pretrained(
            AutoProcessor.from_pretrained,
            self.model_name, **processor_kwargs
        )
        
        tokenizer_kwargs = {
            "trust_remote_code": True,
        }
        if not is_local_path:
            tokenizer_kwargs["token"] = os.getenv("HF_TOKEN")
            
        self.tokenizer = _safe_from_pretrained(
            AutoTokenizer.from_pretrained,
            self.model_name, **tokenizer_kwargs
        )
        print(f"Loaded tokenizer and processor for {self.model_name}")

        # Load the vLLM model
        print(f"Loading model: {self.model_name}")
        optional_kwargs = {"limit_mm_per_prompt": self.limit_mm_per_prompt} if self.limit_mm_per_prompt else {}

        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": True,
            "max_model_len": self.max_model_len,
            "hf_token": os.getenv("HF_TOKEN"),
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "swap_space": 0,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 8192,
            **optional_kwargs,
        }

        if "disable_mm_preprocessor_cache" in inspect.signature(vllm.LLM.__init__).parameters:
            llm_kwargs["disable_mm_preprocessor_cache"] = True

        # Avoid flashinfer JIT GDN kernels that can require newer libstdc++ at runtime.
        if "gdn_prefill_backend" in inspect.signature(EngineArgs.__init__).parameters:
            llm_kwargs["gdn_prefill_backend"] = os.getenv("VLLM_GDN_PREFILL_BACKEND", "triton")

        self.model = vllm.LLM(**llm_kwargs)
        print("✅ vLLM initialized for Qwen3-VL")

    def _process_vision_info(self, messages: List[Dict[str, Any]]) -> List[Image.Image]:
        """
        Extract and preprocess images from messages.
        Resizes images if they exceed the maximum dimensions.
        """
        images = []
        max_size = 1024  # Qwen3-VL supports larger images

        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    image = content["image"].convert("RGB")

                    # Resize if needed while maintaining aspect ratio
                    if image.width > max_size or image.height > max_size:
                        if image.width > image.height:
                            new_width = max_size
                            new_height = int(image.height * (max_size / image.width))
                        else:
                            new_height = max_size
                            new_width = int(image.width * (max_size / image.height))
                        image = image.resize((new_width, new_height))

                    images.append(image)
        return images

    def _format_prompt_for_qwen3vl(self, content: list[dict]) -> TextPrompt:
        """
        Format the prompt for Qwen3-VL with interleaved text and images.

        Args:
            content: List of content items, each with "type" (text/image) and content

        Returns:
            TextPrompt object for vLLM
        """
        # Build messages list
        if self.system_prompt is not None:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": content},
            ]
        else:
            messages = [{"role": "user", "content": content}]

        # Extract images
        images = self._process_vision_info(messages)
        multi_modal_data = {"image": images} if images else {}

        # Apply chat template
        try:
            formatted_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older transformers may not support enable_thinking.
            formatted_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        text_prompt = TextPrompt(prompt=formatted_prompt, multi_modal_data=multi_modal_data)

        return text_prompt

    def generate_one_pass(self, prompt: str) -> List[str]:
        """Single pass generation - not implemented for vLLM, use generate_batch_pass."""
        raise NotImplementedError("generate_one_pass is not implemented for vllm. Use generate_batch_pass instead.")

    def generate_batch_pass(self, g_dict, args: Namespace = None) -> None:
        """
        Generate responses for a batch of prompts using vLLM.
        Responses are added to g_dict in-place.

        Args:
            g_dict: Dictionary with example IDs as keys and data dicts as values.
                    Each data dict must contain "content" (list of text/image items).
            args: Optional Namespace with generation parameters
        """
        print(f"Generating batch pass using vLLM Qwen3-VL with {len(g_dict)} examples")

        # Format prompts
        prompts_to_process = []
        eids_order = []
        start_time = time.time()

        for eid, g_data_item in g_dict.items():
            formatted_prompt = self._format_prompt_for_qwen3vl(g_data_item["content"])
            prompts_to_process.append(formatted_prompt)
            eids_order.append(eid)

        end_time = time.time()
        print(f"Time taken to format {len(prompts_to_process)} prompts: {end_time - start_time:.2f} seconds")

        # Print first example for debugging
        print(f"First example prompt: {prompts_to_process[0]}")

        # Generate using vLLM
        sampling_params = self._get_vllm_sampling_params(args)
        llm_outputs: List[vllm.RequestOutput] = self.model.generate(prompts_to_process, sampling_params)
        llm_outputs = [[generation.text.strip() for generation in output.outputs] for output in llm_outputs]

        # Add generations to g_dict
        for i, generations in enumerate(llm_outputs):
            original_eid = eids_order[i]
            g_dict[original_eid]["generations"] = generations

            # Print first and last few examples for debugging
            if i < 3 or i > len(llm_outputs) - 3:
                print(f"\nExample {original_eid}:")
                for j, generation in enumerate(generations):
                    print(f"  Generation {j}: {generation[:200]}...")

        # Clean up images from g_dict to save memory
        for eid in eids_order:
            if "content" in g_dict[eid] and g_dict[eid]["content"] is not None:
                content_without_images = []
                for content_item in g_dict[eid]["content"]:
                    if content_item["type"] == "text":
                        content_without_images.append(content_item)
                    elif content_item["type"] == "image":
                        content_without_images.append({"type": "image", "text": "IMAGE"})
                g_dict[eid]["content"] = content_without_images
        
        # Clean up local variables and free memory
        del prompts_to_process, llm_outputs, eids_order
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate_llm_as_a_judge(self, batch_requests: List[Dict]):
        """
        Evaluate using LLM-as-a-judge for multimodal tabular reasoning.

        Args:
            batch_requests: List of requests containing:
                - oracle_answer_list: Expected answers
                - our_answer: Model's prediction
                - question: The question
                - page_title, section_title: Table metadata

        Returns:
            List of scores (0 or 1)
        """
        print(f"Processing {len(batch_requests)} LLM-as-a-judge evaluations...")

        # Create evaluation prompts
        evaluation_dict = {}
        for i, request in enumerate(batch_requests):
            oracle_answer = request["oracle_answer_list"][0]
            our_answer = request["our_answer"]
            question = request["question"]
            page_title = request["page_title"]
            section_title = request["section_title"]

            table_metadata = (
                f"Table related to {section_title} in context of {page_title}."
                if section_title and section_title != ""
                else f"Table in context of {page_title}."
            )

            judge_prompt = self._create_llm_judge_prompt(question, oracle_answer, our_answer, table_metadata)
            content = [{"type": "text", "text": judge_prompt}]

            evaluation_dict[str(i)] = {
                "content": content,
                "oracle_answer": oracle_answer,
                "our_answer": our_answer,
                "question": question,
            }

        # Generate judgments
        sampling_args = type(
            "Args",
            (),
            {
                "sampling_n": 1,
                "temperature": 0.1,
                "top_p": 1.0,
                "max_generation_tokens": 2048,
            },
        )()

        self.generate_batch_pass(evaluation_dict, sampling_args)

        # Parse judgments
        scores = []
        for i in range(len(batch_requests)):
            judgment = evaluation_dict[str(i)]["generations"][0].strip().lower()

            correct_index = judgment.rfind("[[correct]]")
            incorrect_index = judgment.rfind("[[incorrect]]")

            if correct_index != -1 or incorrect_index != -1:
                score = 1 if correct_index > incorrect_index else 0
            else:
                print(f"⚠️ Unclear judgment for example {i} (defaulting to incorrect): '{judgment[:100]}'")
                score = 0

            scores.append(score)

            if i < 5 or i >= len(batch_requests) - 5:
                oracle = batch_requests[i]["oracle_answer_list"][0]
                our_ans = batch_requests[i]["our_answer"]
                print(f"Example {i}: Oracle='{oracle}', Our='{our_ans}', Score={score}")

        return scores
