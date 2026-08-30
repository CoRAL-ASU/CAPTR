"""
Generate outputs.
"""

import csv
import datetime
import json
import os
import tempfile
import time
from typing import List, Tuple, Union

import backoff
import openai
from dotenv import load_dotenv
from openai import AzureOpenAI

from generation.generator import Generator
from generation.prompt import PromptBuilder


load_dotenv(".env")


class GPTGenerator(Generator):
    """
    GPT generation wrapper.
    """

    def __init__(self, args):
        self.args = args
        self.current_key_id = 0

        # if the args provided, will initialize with the prompt builder for full usage
        self.prompt_builder = PromptBuilder(args) if args else None

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-10-21",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_time=60,
    )
    def completions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def generate_one_pass(self, prompt: str, verbose: bool = False) -> List[str]:
        """
        Generate one pass with GPT according to the generation phase.
        Returns a list of 2 responses
        """

        print("sending request to OpenAI API")
        result = self._call_openai_api(
            engine=self.args.engine,
            prompt=prompt,
            max_tokens=self.args.max_generation_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.sampling_n,
            stop=self.args.stop_tokens,
        )

        texts = []
        for choice in result["choices"]:
            text = choice.message.content
            texts.append(text)
            if verbose:
                print(text)

        return texts

    def _call_openai_api(
        self,
        engine: str,
        prompt: Union[str, List],
        max_tokens,
        temperature: float,
        top_p: float,
        n: int,
        stop: List[str],
    ):
        start_time = time.time()
        result = None
        while result is None:
            try:
                raw_result = self.completions_with_backoff(
                    model=engine,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "I will give you some x-y examples followed by a x, you need to give me the y, and no other content."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    # api_key=key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                )
                # Keep the raw result for usage logging before potentially modifying it
                result_for_logging = raw_result

                result = {"choices": raw_result.choices}
                print("Openai api inference time:", time.time() - start_time)

                # Log token usage
                try:
                    # Use the result_for_logging which holds the original response object
                    usage = getattr(result_for_logging, "usage", None)
                    if usage:
                        now = datetime.datetime.now()
                        date_str = now.strftime("%Y-%m-%d")
                        log_dir = "consumed_tokens"
                        log_file_path = os.path.join(log_dir, f"{date_str}.csv")

                        # Create directory if it doesn't exist
                        os.makedirs(log_dir, exist_ok=True)

                        # Check if file exists to write header
                        file_exists = os.path.isfile(log_file_path)

                        with open(log_file_path, "a", newline="") as csvfile:
                            fieldnames = ["timestamp", "input_tokens", "output_tokens", "model"]
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                            if not file_exists:
                                writer.writeheader()

                            writer.writerow(
                                {
                                    "timestamp": now.isoformat(),
                                    "input_tokens": usage.prompt_tokens,
                                    "output_tokens": usage.completion_tokens,
                                    "model": engine,
                                }
                            )
                    else:
                        print("Warning: Token usage information not found in API response.")

                except Exception as e:
                    print(f"Error logging token usage: {e}")

                return result
            except Exception:
                import traceback

                traceback.print_exc()

    def generate_batch_pass(self, g_dict) -> None:
        """
        Generate using GPT Batch API. Responses are added to the g_dict in-place.

        Args:
            g_dict: Dict like we have it in our generation pipeline in each step, i.e.: {<eid1>: {"prompt": <prompt1>, "ori_data_item": ..., ...}, <eid2>: ...}
            batch_id: Optional. If you aborted the script at some point and thus already have a batch ID, you can pass it here to resume the batch instead of creating a new one.

        """
        raise NotImplementedError(
            "Haven't yet checked if batch API implementation is correct. We shifted towards using vLLM and thus never tried this batch API."
        )
        # 0. Check that the g_dict follows the expected format (has prompt for each example)
        # for eid, g_data_item in g_dict.items():
        #     assert "prompt" in g_data_item, f"g_dict[{eid}] does not have key 'prompt'"

        # # 1. Extract the prompts & send them to the batch API (if not done before)
        # if batch_id is None:
        #     prompts = []
        #     for eid, g_data_item in g_dict.items():
        #         prompts.append((eid, g_data_item["prompt"]))
        #     batch_id = self._create_batch(prompts)

        # # 2. Get the generations
        # # 2.1 Poll the batch job until it's done
        # batch_output = self._poll_batch_results(batch_id)

        # # 2.2 Parse the batch response
        # response = {}
        # for response_line in batch_output:
        #     response[int(response_line["custom_id"])] = [
        #         choice["message"]["content"] for choice in response_line["response"]["body"]["choices"]
        #     ]

        # # 3. Add the generations to the g_dict
        # for eid, g_pairs in response.items():
        #     g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
        #     g_dict[eid]["generations"] = g_pairs

    def _create_batch(self, prompts: List[Tuple[int, str]]):
        """
        Create a batch job with GPT according to the generation phase.
        """
        batch = []
        for i, (prompt_id, prompt) in enumerate(prompts):
            # Format each prompt as a chat completion request
            request = {
                "custom_id": str(prompt_id),
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self.args.engine,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "I will give you some x-y examples followed by a x, you need to give me the y, and no other content."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": self.args.max_generation_tokens,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "n": self.args.sampling_n,
                    "stop": self.args.stop_tokens,
                },
            }
            batch.append(request)

        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as temp_file:
            for item in batch:
                json.dump(item, temp_file)
                temp_file.write("\n")

            temp_file_path = temp_file.name

        try:
            # Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?tabs=global-batch%2Cstandard-input%2Cpython-key&pivots=programming-language-python
            # Upload the file with a purpose of "batch"
            print(f"Uploading the file... ({temp_file_path})")
            file = self.client.files.create(file=open(temp_file_path, "rb"), purpose="batch")

            print(file.model_dump_json(indent=2))
            file_id = file.id

            print("\nSubmitting a batch job with the file...")
            batch_response = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/chat/completions",
                completion_window="24h",
            )

            batch_id = batch_response.id

            print(f"Batch ID: {batch_id}")

            print(batch_response.model_dump_json(indent=2))

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return batch_id

    def _poll_batch_results(self, batch_id: str):
        batch_response = self.client.batches.retrieve(batch_id)
        status = batch_response.status

        if status not in ("completed", "failed", "canceled"):
            print(
                "The batch job has been started. We will poll it every minute until it's done. You may leave this script running. However, batch jobs may take up to 24h. So you can also abort the script and once the batch is finished, you may re-run the script with --batch_id speicified."
            )

        while status not in ("completed", "failed", "canceled"):
            time.sleep(60)
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status
            print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

        if batch_response.status == "failed":
            for error in batch_response.errors.data:
                print(f"Error code {error.code} Message {error.message}")

            raise Exception("Batch job failed")

        if batch_response.status == "canceled":
            raise Exception("Batch job canceled")

        output_file_id = batch_response.output_file_id

        file_response = self.client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split("\n")

        batch_output = []
        for line in raw_responses:
            batch_output.append(json.loads(line))

        return batch_output
