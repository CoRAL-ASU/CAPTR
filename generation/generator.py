from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict, List, Tuple

from vllm import SamplingParams

from generation.prompt import PromptBuilder


class Generator(ABC):
    """
    Abstract base class for text generation.
    """

    def __init__(self, args):
        self.args = args
        self.current_key_id = 0
        # if the args provided, will initialize with the prompt builder for full usage
        self.prompt_builder = PromptBuilder(args) if args else None

    @abstractmethod
    def generate_one_pass(self, prompt: str, verbose: bool = False) -> List[str]:
        """
        Generate one pass according to the generation phase.
        Returns a list of responses.
        """
        pass

    @abstractmethod
    def generate_batch_pass(self, g_dict, args: Namespace) -> None:
        """
        Generate using batch API. Responses are added to the g_dict in-place.

        Args:
            g_dict: Dict like we have it in our generation pipeline in each step, i.e.: {<eid1>: {"prompt": <prompt1>, "ori_data_item": ..., ...}, <eid2>: ...}
        """
        pass

    @abstractmethod
    def evaluate_llm_as_a_judge(self, batch_requests: List[Dict]):
        """
        Evaluate the LLM as a judge.
        Args:
            batch_requests: List of requests to evaluate. Must contain the following keys:
                - oracle_answer_list: List of oracle answers.
                - our_answer: Our answer.
                - image: Image that replaced the image placeholder.
                - question: Question.
            Returns:
                List of scores. Each score is either 0 or 1.
        """
        pass

    # Helper function for vllm generation
    def _get_vllm_sampling_params(self, args=None):
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

        return SamplingParams(
            n=getattr(args, "sampling_n", 1),
            temperature=getattr(args, "temperature", 0.7),
            top_p=getattr(args, "top_p", 0.95),
            max_tokens=getattr(args, "max_generation_tokens", 512),
            # stop=stop_tokens,
        )

    # Helper function for LLM-as-a-judge generation
    def _create_llm_judge_prompt(self, question: str, oracle_answer: str, our_answer: str, table_metadata: str) -> str:
        """
        Creates a prompt for LLM-as-a-judge to evaluate the correctness
        of a generated answer against an oracle answer in a multimodal tabular reasoning context,
        focusing on entity equivalence.
        """

        prompt = f"""You are an impartial, precise, and expert judge. Your task is to evaluate the factual correctness of a model's response against an oracle (ground truth) answer in the context of a question asked about a table.

# Background Information
The model you are evaluating answers questions based on a table where some cells contain images (e.g., logos, photos) instead of text. The Oracle Answer represents the ground truth entity or fact that was originally in the cell. The model must interpret the image to identify this information.

Crucially, the Oracle Answer is just *one possible textual realization* of the underlying information. Your goal is to determine if the Model Answer correctly identifies the *same underlying entity or fact*.

# Input Data
Table Metadata: "{table_metadata}"
Question: "{question}"
Oracle Answer (Ground Truth): "{oracle_answer}"
Model Answer (Prediction): "{our_answer}"

# Evaluation Criteria
You must judge if the Model Answer is CORRECT or INCORRECT.

The prediction is CORRECT if and only if:
1.  **Factual Accuracy:** It correctly answers the Question based on the ground truth.
2.  **Equivalence:** It identifies the exact same entity, value, or fact as the Oracle Answer.

# Guidelines for Equivalence and Specificity
1.  **Named Entities (Organizations, People, Locations, etc.):** The Model Answer does not need to match the specificity of the Oracle Answer, as long as it correctly and unambiguously identifies the entity.
    *   **Naming Variations:** Accept synonyms, acronyms, full names, or widely recognized alternative names. (e.g., "IBM" vs. "International Business Machines").
    *   **Over-specification:** If the Model Answer is more specific than the Oracle but refers to the same entity, it is CORRECT.
        *   *Example:* Oracle Answer: "Darmstadt", Model Answer: "Technical University of Darmstadt" -> CORRECT.
    *   **Under-specification:** If the Model Answer is less specific than the Oracle but still unambiguously identifies the entity in context, it is CORRECT.
        *   *Example:* Oracle Answer: "Technical University of Darmstadt", Model Answer: "Darmstadt" -> CORRECT.
2.  **Numerical and Date Information:** For numbers, quantities, or dates, the value must be exact, although formatting can vary (e.g., "1,000" vs "1000"; "Jan 1st" vs "January 1").
3.  **Formatting and Phrasing:** Minor differences in punctuation or phrasing are acceptable if the meaning is preserved.

The prediction is INCORRECT if:
1.  It identifies a different entity or value than the Oracle Answer (e.g., confusing "Darmstadt University of Applied Sciences" with "Technical University of Darmstadt").
2.  It includes extraneous facts or hallucinations unrelated to the identification of the entity or fact.
3.  It is too ambiguous or general to determine the entity (e.g., answering "A university" when a specific name is required).

# Instructions
1.  Analyze the Question and Oracle Answer to identify the core entity or fact(s) required.
2.  Compare the Model Answer with the Oracle Answer based on the Guidelines for Equivalence and Specificity.
3.  Provide a detailed, step-by-step Chain-of-Thought (CoT) explaining your reasoning. Explicitly address any naming or specificity variations and explain why they are acceptable or unacceptable realizations of the ground truth.
4.  Conclude with your final verdict in the exact format: "[[CORRECT]]" or "[[INCORRECT]]".

Evaluation:
"""

        return prompt

    # ######################################################### prompt creation functions #########################################################
    def prompt_row_truncate(
        self,
        prompt: str,
        num_rows_to_remain: int,
        table_end_token: str = "*/",
    ):
        """
        Fit prompt into max token limits by row truncation.
        """
        table_end_pos = prompt.rfind(table_end_token)
        assert table_end_pos != -1
        prompt_part1, prompt_part2 = prompt[:table_end_pos], prompt[table_end_pos:]
        prompt_part1_lines = prompt_part1.split("\n")[::-1]
        trunc_line_index = None
        for idx, line in enumerate(prompt_part1_lines):
            if "\t" not in line:
                continue
            row_id = int(line.split("\t")[0])
            if row_id <= num_rows_to_remain:
                trunc_line_index = idx
                break
        new_prompt_part1 = "\n".join(prompt_part1_lines[trunc_line_index:][::-1])
        prompt = new_prompt_part1 + "\n" + prompt_part2
        return prompt

    def build_few_shot_prompt_from_file(self, file_path: str, n_shots: int):
        """
        Build few-shot prompt for generation from file.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        few_shot_prompt_list = []
        one_shot_prompt = ""
        last_line = None
        for line in lines:
            if line == "\n" and last_line == "\n":
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ""
            else:
                one_shot_prompt += line
            last_line = line
        few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[:n_shots]
        few_shot_prompt_list[-1] = few_shot_prompt_list[
            -1
        ].strip()  # It is essential for prompting to remove extra '\n'
        few_shot_prompt = "\n".join(few_shot_prompt_list)
        return few_shot_prompt

    def build_generate_prompt(self, data_item: Dict, generate_type: Tuple, args: Namespace, **kwargs):
        """
        Build the generate prompt
        """
        prompt_builder = PromptBuilder(args)
        return prompt_builder.build_generate_prompt(**data_item, generate_type=generate_type, **kwargs)
