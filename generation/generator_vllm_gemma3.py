# Important: Big bottleneck is image preprocessing. We can't use CUDA for this however, since we would need to run this on a different GPU than vllm..
# If we use the same GPU for image preprocessing and model inference, this leads to 10x slower performance by vllm (likely due to GPU fragmentation).
# Hence we opt to use CPU.
# One not-yet explored option is to preprocess the images, i.e. use the vision tower and multi modal projector, and then use the GPU. afterwards we could init vllm.
# I already implemented some things for this, but not yet tried it.
import os
import time
import math
import inspect
from argparse import Namespace
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from PIL import Image

from generation.generator import Generator
from transformers import AutoProcessor, AutoTokenizer, Gemma3Processor


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


# ##### patch vllm to use_fast=True #####
if not hasattr(Gemma3Processor, "_vllm_fast_patched"):
    original_from_pretrained = Gemma3Processor.from_pretrained

    @classmethod
    def fast_from_pretrained(cls, *args, **kwargs):
        # Always set use_fast=True unless explicitly disabled
        kwargs.setdefault("use_fast", True)
        return original_from_pretrained(*args, **kwargs)

    Gemma3Processor.from_pretrained = fast_from_pretrained
    Gemma3Processor._vllm_fast_patched = True
    print("✅ Gemma3 fast processor patch applied")

# Then import vllm
import vllm
from vllm import EngineArgs, TextPrompt


load_dotenv(".env")


class VLLMGeneratorGemma3(Generator):
    """
    vllm generation wrapper for gemma3 models.
    """

    def __init__(
        self,
        args,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.9,
        system_prompt="I will give you some x-y examples followed by a x, you need to give me the y, and no other content.",
    ):
        """
        Args:
            args: ArgumentParser object
            limit_mm_per_prompt: Optional. THIS WILL BE SET TO NONE FOR TEXT-ONLY MODELS. If you are using a model that supports multiple modalities as interleaved input, you can pass the limit_mm_per_prompt here.
        """
        print(f"Initializing VLLMGenerator with args: {args}")
        assert args.engine == "vllm", "Generator only supports vllm engine"
        assert args.vllm_model_name, "vllm_model_name is required when engine is vllm"
        assert args.number_of_gpus, "number_of_gpus is required when engine is vllm"
        self.args = args

        # additional args
        self.model_name = args.vllm_model_name
        self.tensor_parallel_size = args.number_of_gpus
        self.max_model_len = args.max_api_total_tokens
        self.prompt_mode = (args.prompt_mode if args.prompt_mode else "text-only")  # omni chat template is a bit different (list instead of string) and needs a specific system prompt.  # fmt: skip
        assert self.prompt_mode == "text-image", "prompt_mode must be 'text-image'"

        print(f"torch.get_num_threads(): {torch.get_num_threads()}")

        # set important variables
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.system_prompt = system_prompt
        self.max_image_long_edge = 896
        self.min_image_edge = 128
        # initialize model
        self.gpu_memory_utilization = gpu_memory_utilization
        self._load_model()
        # self._load_vision_tower()

    # ######################################################### vllm model functions #########################################################
    def _load_model(self):
        # Debug: Check GPU visibility
        import os
        print(f"DEBUG _load_model: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
        print(f"DEBUG _load_model: torch.cuda.device_count() = {torch.cuda.device_count()}")
        print(f"DEBUG _load_model: tensor_parallel_size = {self.tensor_parallel_size}")
        
        # Detect if model_name is a local path
        is_local_path = os.path.isdir(self.model_name)
        
        processor_kwargs = {
            "trust_remote_code": True,
            "use_fast": True,
        }
        if not is_local_path:
            processor_kwargs["token"] = os.getenv("HF_TOKEN")
            
        self.processor = _safe_from_pretrained(
            AutoProcessor.from_pretrained,
            self.model_name, **processor_kwargs
        )

        tokenizer_kwargs = {}
        if not is_local_path:
            tokenizer_kwargs["token"] = os.getenv("HF_TOKEN")
            
        self.tokenizer = _safe_from_pretrained(AutoTokenizer.from_pretrained, self.model_name, **tokenizer_kwargs)

        print(f"loaded tokenizer for {self.model_name}")

        print(f"Loading model: {self.model_name}")
        optional_kwargs = {"limit_mm_per_prompt": self.limit_mm_per_prompt} if self.limit_mm_per_prompt else {}
        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": True,
            "max_model_len": self.max_model_len,
            "hf_token": os.getenv("HF_TOKEN"),
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "swap_space": 4,
            "mm_processor_kwargs": {
                "do_pan_and_scan": False,
            },
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "enforce_eager": False,
            **optional_kwargs,
        }

        # Keep backward compatibility across vLLM versions where this arg may not exist.
        if "disable_mm_preprocessor_cache" in inspect.signature(vllm.LLM.__init__).parameters:
            llm_kwargs["disable_mm_preprocessor_cache"] = True

        # Avoid flashinfer JIT GDN kernels that can require newer libstdc++ at runtime.
        if "gdn_prefill_backend" in inspect.signature(EngineArgs.__init__).parameters:
            llm_kwargs["gdn_prefill_backend"] = os.getenv("VLLM_GDN_PREFILL_BACKEND", "triton")

        self.gemma3_model = vllm.LLM(
            **llm_kwargs,
        )

    # ######################################################### formatting functions #########################################################
    def _adaptive_image_area_budget(self, num_images: int) -> float:
        """
        Return target image area budget per image.
        Fewer images can keep larger resolution; many images are downscaled more aggressively.
        """
        base_area = float(self.max_image_long_edge * self.max_image_long_edge)
        if num_images <= 4:
            scale = 1.0
        elif num_images <= 8:
            scale = 0.75
        elif num_images <= 16:
            scale = 0.55
        elif num_images <= 32:
            scale = 0.40
        else:
            scale = 0.30
        return base_area * scale

    def _resize_image_adaptive(self, image: Image.Image, target_area: float) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        aspect_ratio = max(width, height) / max(1, min(width, height))

        # For very elongated images, reduce area a bit more to avoid oversized feature maps.
        if aspect_ratio >= 3.0:
            target_area *= 0.8
        elif aspect_ratio >= 2.0:
            target_area *= 0.9

        # Also enforce an upper bound on long edge.
        long_edge = max(width, height)
        long_edge_scale = min(1.0, self.max_image_long_edge / float(long_edge))

        area = float(width * height)
        area_scale = min(1.0, math.sqrt(target_area / area)) if area > 0 else 1.0
        resize_scale = min(long_edge_scale, area_scale)

        if resize_scale >= 0.999:
            return image

        new_width = max(self.min_image_edge, int(width * resize_scale))
        new_height = max(self.min_image_edge, int(height * resize_scale))

        return image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    def _process_vision_info_gemma3(self, messages: List[Dict[str, Any]]) -> List[Image.Image]:
        """Extract and adaptively resize images from the message."""
        raw_images = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    raw_images.append(content["image"].convert("RGB"))

        if not raw_images:
            return []

        target_area = self._adaptive_image_area_budget(len(raw_images))
        images = [self._resize_image_adaptive(image, target_area) for image in raw_images]
        return images

    def _format_prompt_for_gemma3_image(self, content: list[dict]) -> TextPrompt:
        """
        This function formats the prompt for vllm models that support multiple modalities as interleaved input.
        """
        if self.system_prompt is not None:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": content},
            ]
        else:
            messages = [{"role": "user", "content": content}]

        images = self._process_vision_info_gemma3(messages)
        multi_modal_data = {"image": images} if images != [] else {}
        formatted_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        text_prompt = TextPrompt(prompt=formatted_prompt, multi_modal_data=multi_modal_data)

        return text_prompt

    # ######################################################### generation functions #########################################################
    def generate_one_pass(self, prompt: str) -> List[str]:
        raise NotImplementedError("generate_one_pass is not implemented for vllm. Use generate_batch_pass instead.")

    def generate_batch_pass(self, g_dict, args: Namespace = None) -> None:
        """
        Generate using GPT Batch API. Responses are added to the g_dict in-place.

        Args:
            g_dict: Dict like we have it in our generation pipeline in each step, i.e.:
                1. {<eid1>: {"prompt": str, "ori_data_item": ..., ...}, <eid2>: ...}
                2. {<eid1>: {"content": list[dict], "ori_data_item": ..., ...}, <eid2>: ...}
        """
        print(f"Generating batch pass using vllm with {len(g_dict)} examples")

        # 1. Extract and format prompts
        prompts_to_process = []
        eids_order = []  # To map results back to the original eids
        start_time = time.time()
        for eid, g_data_item in g_dict.items():
            formatted_prompt = self._format_prompt_for_gemma3_image(g_data_item["content"])

            prompts_to_process.append(formatted_prompt)
            eids_order.append(eid)

        end_time = time.time()
        print(f"Time taken to format {len(prompts_to_process)} prompts: {end_time - start_time} seconds")

        # for the first example, print the formatted prompt
        print(f"First example: {prompts_to_process[0]}")

        # 2. Get the generations using vLLM
        sampling_params = self._get_vllm_sampling_params(args)
        llm_outputs: List[vllm.RequestOutput] = self.gemma3_model.generate(prompts_to_process, sampling_params)
        llm_outputs = [[generation.text.strip() for generation in output.outputs] for output in llm_outputs]

        # 3. Add the generations to the g_dict
        for i, generations in enumerate(llm_outputs):
            original_eid = eids_order[i]
            g_dict[original_eid]["generations"] = generations

            # For the first 3 and last 3 examples, print the generations
            if i < 3 or i > len(llm_outputs) - 3:
                print(f"\nExample {original_eid}:")
                for i, generation in enumerate(generations):
                    print(f"  Generation {i}: {generation}")

        # 4. Remove images from g_dict in-place to save memory
        for eid in eids_order:
            if "content" in g_dict[eid] and g_dict[eid]["content"] is not None:
                content_without_images = []
                for content_item in g_dict[eid]["content"]:
                    if content_item["type"] == "text":
                        content_without_images.append(content_item)
                    elif content_item["type"] == "image":
                        # The PIL.Image object is not json serializable. So we replace it with a placeholder.
                        content_without_images.append({"type": "image", "text": "IMAGE"})
                g_dict[eid]["content"] = content_without_images

    def evaluate_llm_as_a_judge(self, batch_requests: List[Dict]):
        """
        Evaluate the LLM as a judge for multimodal tabular reasoning.

        This function determines if the model's answer correctly describes/identifies
        the image content in the context of the question and expected answer.

        Args:
            batch_requests: List of requests to evaluate. Must contain the following keys:
                - oracle_answer_list: List of oracle answers (original text that was replaced by image).
                - our_answer: Our model's answer (description of the image).
                - image: PIL Image that replaced the image placeholder.
                - question: The question being asked.

        Returns:
            List of scores. Each score is either 0 or 1.
        """
        print(f"Processing {len(batch_requests)} LLM-as-a-judge evaluations...")

        # Create evaluation prompts for batch processing
        evaluation_dict = {}
        for i, request in enumerate(batch_requests):
            oracle_answer = request["oracle_answer_list"][0]  # Take first oracle answer
            our_answer = request["our_answer"]
            # image = request["image"]
            question = request["question"]
            page_title = request["page_title"]
            section_title = request["section_title"]  # fmt: skip

            table_metadata = (
                f"Table related to {section_title} in context of {page_title}."
                if section_title and section_title != ""
                else f"Table in context of {page_title}."
            )

            # Create the LLM-as-a-judge prompt
            judge_prompt = self._create_llm_judge_prompt(question, oracle_answer, our_answer, table_metadata)

            # Format as multimodal content for the judge
            content = [{"type": "text", "text": judge_prompt}]

            evaluation_dict[str(i)] = {
                "content": content,
                "oracle_answer": oracle_answer,
                "our_answer": our_answer,
                "question": question,
            }

        # Generate judgments using the model
        sampling_args = type(
            "Args",
            (),
            {
                "sampling_n": 1,
                "temperature": 0.1,  # temperature follows: https://aclanthology.org/2025.acl-long.252/
                "top_p": 1.0,
                "max_generation_tokens": 2048,
            },
        )()

        self.generate_batch_pass(evaluation_dict, sampling_args)

        # Parse the judgments and return scores
        scores = []
        for i in range(len(batch_requests)):
            judgment = evaluation_dict[str(i)]["generations"][0].strip().lower()

            # Parse the judgment: look for the last occurrence of "[[CORRECT]]" or "[[INCORRECT]]"
            correct_index = judgment.rfind("[[correct]]")  # lowercase
            incorrect_index = judgment.rfind("[[incorrect]]")  # lowercase
            if correct_index != -1 or incorrect_index != -1:
                score = 1 if correct_index > incorrect_index else 0
            else:
                print(
                    f"⚠️  Unclear judgment for example {i} (defaulting to incorrect). Neither [[correct]] nor [[incorrect]] found in: '{judgment}'"
                )
                score = 0

            scores.append(score)

            # Print some examples for debugging
            if i < 5 or i >= len(batch_requests) - 5:
                oracle = batch_requests[i]["oracle_answer_list"][0]
                our_ans = batch_requests[i]["our_answer"]
                print(f"Example {i}: Oracle='{oracle}', Our='{our_ans}', Judgment='{judgment}' (Score={score})")

        return scores
