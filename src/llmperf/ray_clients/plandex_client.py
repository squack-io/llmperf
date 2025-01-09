import json
import os
import time
from typing import Any, Dict

import ray
import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class PlandexClient(LLMClient):
    """Client for Plandex API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt_len = 36003
        messages = [{"role": "user", "content": "Hello! Can you tell me a quick joke?"}]
        model = request_config.model
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {common_metrics.ERROR_CODE: None, common_metrics.ERROR_MSG: ""}

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        # API configuration
        address = os.environ.get("OPENAI_API_BASE")
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("the environment variable OPENAI_API_KEY must be set.")

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"

        try:
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=180,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        # Convert bytes to string if necessary
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")

                        # Skip OpenRouter processing messages
                        if line.strip() == ": OPENROUTER PROCESSING":
                            continue

                        # Handle SSE format
                        if line.startswith("data: "):
                            line = line[6:]  # Remove 'data: ' prefix
                        else:
                            continue  # Skip non-data lines

                        # Skip [DONE] messages
                        if line.strip() == "[DONE]":
                            continue

                        # Parse JSON data
                        data = json.loads(line)

                        if "error" in data:
                            error_msg = data["error"]["message"]
                            error_response_code = data["error"]["code"]
                            raise RuntimeError(data["error"]["message"])

                        # Extract content from the choices
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                tokens_received += 1

                                if not ttft:
                                    ttft = time.monotonic() - start_time
                                    time_to_next_token.append(ttft)
                                else:
                                    time_to_next_token.append(
                                        time.monotonic()
                                        - most_recent_received_token_time
                                    )

                                most_recent_received_token_time = time.monotonic()
                                generated_text += content

                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {line}")
                        print(f"Error: {e}")
                        continue

            total_request_time = time.monotonic() - start_time
            output_throughput = (
                tokens_received / total_request_time if total_request_time > 0 else 0
            )

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(f"Error Response Code: {error_response_code}")
            return metrics, generated_text, request_config

        # Update final metrics
        metrics.update(
            {
                common_metrics.INTER_TOKEN_LAT: sum(time_to_next_token),
                common_metrics.TTFT: ttft,
                common_metrics.E2E_LAT: total_request_time,
                common_metrics.REQ_OUTPUT_THROUGHPUT: output_throughput,
                common_metrics.NUM_TOTAL_TOKENS: tokens_received + prompt_len,
                common_metrics.NUM_OUTPUT_TOKENS: tokens_received,
                common_metrics.NUM_INPUT_TOKENS: prompt_len,
            }
        )

        return metrics, generated_text, request_config
