import math
import os
from argparse import Namespace
from typing import Dict, List, Tuple

import vllm
from dotenv import load_dotenv
from PIL import Image
from qwen_omni_utils import process_mm_info
from vllm import TextPrompt

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


class VLLMGeneratorQwen(Generator):
    """
    vllm generation wrapper for Qwen models. Should work for other models too, but for now we only use Qwen in our experiments.
    """

    def __init__(self, args, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0}, gpu_memory_utilization=0.8):
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
        self.max_model_len = args.max_api_total_tokens + 200  # some buffer
        raw_prompt_mode = (args.prompt_mode if args.prompt_mode else "text-only")
        if raw_prompt_mode == "text-image":
            raw_prompt_mode = "omni"
        self.prompt_mode = raw_prompt_mode  # omni chat template is a bit different (list instead of string) and needs a specific system prompt.  # fmt: skip
        assert self.prompt_mode in ["text-only", "omni"], "prompt_mode must be one of: 'text-only', 'omni', 'text-image'"

        # set important variables
        if self.prompt_mode == "omni":
            self.processor = None  # Will be loaded in _load_model
            self.tokenizer = None
            self.limit_mm_per_prompt = limit_mm_per_prompt
            self.system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        else:
            self.processor = None
            self.tokenizer = None  # Will be loaded in _load_model
            self.limit_mm_per_prompt = None
            self.system_prompt = (
                "I will give you some x-y examples followed by a x, you need to give me the y, and no other content."
            )

        # initialize model
        self.gpu_memory_utilization = gpu_memory_utilization
        self._load_model()

    # ######################################################### vllm model functions #########################################################
    def _load_model(self):
        # Detect if model_name is a local path
        is_local_path = os.path.isdir(self.model_name)
        
        if self.prompt_mode == "omni":
            print("Loading AutoProcessor & tokenizer for omni model")
            processor_kwargs = {"trust_remote_code": True}
            if not is_local_path:
                processor_kwargs["token"] = os.getenv("HF_TOKEN")
            self.processor = _safe_from_pretrained(AutoProcessor.from_pretrained, self.model_name, **processor_kwargs)
            
        tokenizer_kwargs = {}
        if not is_local_path:
            tokenizer_kwargs["token"] = os.getenv("HF_TOKEN")
            
        print("Loading AutoTokenizer")
        self.tokenizer = _safe_from_pretrained(AutoTokenizer.from_pretrained, self.model_name, **tokenizer_kwargs)

        print(f"Loading model: {self.model_name}")
        optional_kwards = {"limit_mm_per_prompt": self.limit_mm_per_prompt} if self.limit_mm_per_prompt else {}
        self.llm = vllm.LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="auto",  # gemma 3 would need bfloat16, since its buggy AFAIK. But we are not using gemma3, so we can use auto.
            trust_remote_code=True,
            hf_token=os.getenv("HF_TOKEN"),
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            **optional_kwards,
        )
        print("✅ vllm initialized.")

    def _get_vllm_sampling_params(self, args) -> vllm.SamplingParams:
        """
        Creates vLLM SamplingParams from self.args.
        Assumes self.args has: max_generation_tokens, temperature, top_p, sampling_n, stop_tokens
        """
        if args is None:
            print("⚠️ ⚠️ Attention: No args passed to _get_vllm_sampling_params. Using self.args instead.")
            args = self.args

        # Check for stop tokens and warn if they exist
        stop_tokens = getattr(args, "stop_tokens", [])
        if stop_tokens:
            print("⚠️ Warning: stop_tokens are provided but not used in vLLM SamplingParams.")

        return vllm.SamplingParams(
            n=getattr(self.args, "sampling_n", 1),
            temperature=getattr(self.args, "temperature", 0.7),
            top_p=getattr(self.args, "top_p", 0.95),
            max_tokens=getattr(self.args, "max_generation_tokens", 512),
            # stop=stop_tokens,
        )

    # ######################################################### formatting functions #########################################################
    def _format_prompt_for_vllm_text(self, prompt: str) -> str:
        """
        E.g. for Qwen3; This function formats the prompt for vllm text-generation models.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return formatted_prompt

    def _calculate_image_tokens(self, image_size: Tuple[int, int]) -> int:
        """
        Calculates the token cost for a single image based on Qwen2.5-Omni rules.
        - Patch size: 14x14
        - Rule: Each image is treated as two identical frames.
        """
        PATCH_SIZE = 14
        FRAME_MULTIPLIER = 2

        width, height = image_size

        # Calculate patches needed, rounding up for non-divisible dimensions
        patches_w = math.ceil(width / PATCH_SIZE)
        patches_h = math.ceil(height / PATCH_SIZE)

        total_patches_per_frame = patches_w * patches_h

        # Apply the "two identical frames" rule
        token_cost = total_patches_per_frame * FRAME_MULTIPLIER

        return token_cost

    def _resize_images(self, content: list[dict]):
        # 1. First Pass: Calculate token cost of all images & token cost of text part of prompt
        total_image_tokens = 0
        total_text_tokens = 0
        image_info = []  # Store tuples of (index, image_object)
        for i, item in enumerate(content):
            if item.get("type") == "image" and "image" in item:
                img = item["image"]
                if isinstance(img, Image.Image):
                    image_info.append((i, img))
                    total_image_tokens += self._calculate_image_tokens(img.size)
            elif item.get("type") == "text":
                total_text_tokens += len(self.tokenizer.encode(item["text"]))

        # 2. calculate max_image_tokens (have some buffer for other tokens)
        max_generation_tokens = getattr(self.args, "max_generation_tokens", 512)
        max_image_tokens = self.args.max_api_total_tokens - total_text_tokens - max_generation_tokens - 1000

        # 3. resize images if needed
        if total_image_tokens > max_image_tokens and total_image_tokens > 0:
            scaling_ratio = max_image_tokens / total_image_tokens
            scaling_factor = math.sqrt(scaling_ratio)

            # Second Pass: Apply resizing
            resized_token_count = 0
            for index, img in image_info:
                original_w, original_h = img.size
                new_w = int(original_w * scaling_factor)
                new_h = int(original_h * scaling_factor)

                # Ensure dimensions are at least 1x1
                new_w = max(1, new_w)
                new_h = max(1, new_h)

                # Resize and replace the image object in the original content list
                # LANCZOS is a high-quality filter for downsampling
                resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                content[index]["image"] = resized_img
                resized_token_count += self._calculate_image_tokens(resized_img.size)

            print(f"⚠️ Total image tokens ({total_image_tokens}) exceed limit ({max_image_tokens}). Resizing images with scaling factor {scaling_factor}⚠️. New estimated image tokens: {resized_token_count}")  # fmt: skip

        return content

    def _format_prompt_for_vllm_omni(self, content: list[dict]) -> TextPrompt:
        """
        E.g. for Qwen2.5-omni; This function formats the prompt for vllm models that support multiple modalities as interleaved input.
        It proactively resizes images if their combined token count exceeds a threshold.
        """
        content = self._resize_images(content)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},  # content may now contain resized images
        ]

        # 1. Get the images in the right format using the utility
        audio, images, video = process_mm_info(messages, use_audio_in_video=True)  # We dont have audio and video.
        mm_data = {}  # "audio", "image", "video"
        if images is not None:
            mm_data["image"] = images

        multi_modal_data = mm_data if mm_data else None

        # 2. Bring the content into the right format
        # For omni, we use the processor to apply the chat template
        formatted_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        if "Qwen2.5-Omni" in self.model_name:
            formatted_prompt = formatted_prompt[0]  # apply_chat_template returns a list for Qwen2.5-Omni

        text_prompt = TextPrompt(prompt=formatted_prompt, multi_modal_data=multi_modal_data)

        return text_prompt

    # ######################################################### generation functions #########################################################
    def generate_one_pass(self, prompt: str) -> List[str]:
        raise NotImplementedError("generate_one_pass is not implemented for vllm. Use generate_batch_pass instead.")

    def generate_batch_pass(self, g_dict, args: Namespace) -> None:
        """
        Generate using GPT Batch API. Responses are added to the g_dict in-place.

        Args:
            g_dict: Dict like we have it in our generation pipeline in each step, i.e.:
                1. {<eid1>: {"prompt": str, "ori_data_item": ..., ...}, <eid2>: ...}
                2. {<eid1>: {"content": list[dict], "ori_data_item": ..., ...}, <eid2>: ...}
        """
        # 0. Check that the g_dict follows the expected format (has prompt for each example)
        print(f"Generating batch pass using vllm with {len(g_dict)} examples")
        for eid, g_data_item in g_dict.items():
            assert "prompt" in g_data_item or "content" in g_data_item, (
                f"g_dict[{eid}] does not have key 'prompt' or 'content'"
            )
            if "prompt" in g_data_item:
                assert self.prompt_mode == "text-only", "prompt_mode must be 'text-only' when 'prompt' is in g_dict"
            elif "content" in g_data_item:
                assert self.prompt_mode == "omni", "prompt_mode must be 'omni' when 'content' is in g_dict"

        # 1. Extract and format prompts
        prompts_to_process = []
        eids_order = []  # To map results back to the original eids
        total_image_items = 0
        prompts_with_images = 0
        for eid, g_data_item in g_dict.items():
            if self.prompt_mode == "text-only":
                formatted_prompt = self._format_prompt_for_vllm_text(g_data_item["prompt"])
            elif self.prompt_mode == "omni":
                content = g_data_item["content"]
                image_items = sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image")
                total_image_items += image_items
                if image_items > 0:
                    prompts_with_images += 1
                formatted_prompt = self._format_prompt_for_vllm_omni(g_data_item["content"])

            prompts_to_process.append(formatted_prompt)
            eids_order.append(eid)

        # for the first example, print the formatted prompt
        print(f"First example: {prompts_to_process[0]}")
        if self.prompt_mode == "omni":
            print(
                f"Multimodal batch summary: prompts with images {prompts_with_images}/{len(prompts_to_process)}; "
                f"total attached image items {total_image_items}."
            )

        # 2. Get the generations using vLLM
        sampling_params = self._get_vllm_sampling_params(args)
        llm_outputs: List[vllm.RequestOutput] = self.llm.generate(prompts_to_process, sampling_params)
        llm_outputs = [[generation.text.strip() for generation in output.outputs] for output in llm_outputs]

        # 3. Add the generations to the g_dict
        for i, generations in enumerate(llm_outputs):
            original_eid = eids_order[i]

            # Sort by cumulative_logprob (the second element of the tuple) in descending order
            # generations = sorted(generations, key=lambda x: x[-1], reverse=True)
            g_dict[original_eid]["generations"] = generations

            # For the first 3 and last 3 examples, print the generations
            if i < 3 or i > len(llm_outputs) - 3:
                print(f"\nExample {original_eid}:")
                for i, generation in enumerate(generations):
                    print(f"  Generation {i}: {generation}")

    def evaluate_llm_as_a_judge(self, batch_requests: List[Dict]):
        """
        Evaluate model outputs with LLM-as-a-judge prompts.
        """
        print(f"Processing {len(batch_requests)} LLM-as-a-judge evaluations...")

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

            if self.prompt_mode == "text-only":
                evaluation_dict[str(i)] = {
                    "prompt": judge_prompt,
                    "oracle_answer": oracle_answer,
                    "our_answer": our_answer,
                    "question": question,
                }
            else:
                evaluation_dict[str(i)] = {
                    "content": [{"type": "text", "text": judge_prompt}],
                    "oracle_answer": oracle_answer,
                    "our_answer": our_answer,
                    "question": question,
                }

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

        scores = []
        for i in range(len(batch_requests)):
            judgment = evaluation_dict[str(i)]["generations"][0].strip().lower()
            correct_index = judgment.rfind("[[correct]]")
            incorrect_index = judgment.rfind("[[incorrect]]")

            if correct_index != -1 or incorrect_index != -1:
                score = 1 if correct_index > incorrect_index else 0
            else:
                print(
                    f"⚠️ Unclear judgment for example {i} (defaulting to incorrect): '{judgment[:100]}'"
                )
                score = 0

            scores.append(score)

            if i < 5 or i >= len(batch_requests) - 5:
                oracle = batch_requests[i]["oracle_answer_list"][0]
                our_ans = batch_requests[i]["our_answer"]
                print(f"Example {i}: Oracle='{oracle}', Our='{our_ans}', Score={score}")

        return scores
