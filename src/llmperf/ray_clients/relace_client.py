import json
import os
import time
from typing import Any, Dict, Tuple

import ray
import requests

from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class RelaceClient:
    """Client for Relace API."""

    def llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[Dict[str, Any], str, RequestConfig]:
        print(f"Starting request with config: {request_config}")
        user_prompt = request_config.user_prompt
        system_prompt = request_config.system_prompt

        message = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]
        print("Messages constructed")
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        print("Request body prepared")
        
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        print(f"Request started at: {start_time}")
        
        address = os.environ.get("OPENAI_API_BASE")
        print(f"Using API base address: {address}")
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("the environment variable OPENAI_API_KEY must be set.")
        print("API key found")
        headers = {"Authorization": f"Bearer {key}"}
        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"
        print(f"Final API endpoint: {address}")
        
        try:
            print("Initiating API request...")
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=180,
                headers=headers,
            ) as response:
                print(f"Response status code: {response.status_code}")
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    print(f"Error response received: {error_msg}")
                    response.raise_for_status()
                print("Processing response stream...")
                for chunk in response.iter_lines(chunk_size=None):
                    chunk = chunk.strip()

                    if not chunk:
                        continue
                    stem = "data: "
                    chunk = chunk[len(stem) :]
                    if chunk == b"[DONE]":
                        print("Received DONE signal")
                        continue
                    tokens_received += 1
                    print(f"Processing chunk {tokens_received}")
                    data = json.loads(chunk)

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        print(f"Error in response data: {error_msg}")
                        raise RuntimeError(data["error"]["message"])

                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        current_time = time.monotonic()
                        if not ttft:
                            ttft = current_time - start_time
                            print(f"Time to first token: {ttft}")
                            time_to_next_token.append(ttft)
                        else:
                            token_latency = current_time - most_recent_received_token_time
                            print(f"Token latency: {token_latency}")
                            time_to_next_token.append(token_latency)
                        most_recent_received_token_time = current_time
                        generated_text += delta["content"]

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time
            print(f"Request completed. Total time: {total_request_time}")
            print(f"Output throughput: {output_throughput} tokens/second")

        except Exception as e:
            print(f"Exception occurred during request: {str(e)}")
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(f"Error response code: {error_response_code}")

        print("Calculating final metrics...")
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(
            time_to_next_token
        )  # This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = (
            tokens_received + request_config.num_input_tokens
        )
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = request_config.num_input_tokens
        print(f"Final metrics: {metrics}")

        return metrics, generated_text, request_config
