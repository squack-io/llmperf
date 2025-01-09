import json
import os
import time
from typing import Any, Dict

import ray
import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@ray.remote
class PlandexClient(LLMClient):
    """Client for Plandex API."""

    def _log_request_details(self, address: str, headers: Dict, body: Dict):
        """Log detailed request information for debugging."""
        # Remove sensitive info from headers
        safe_headers = headers.copy()
        if "Authorization" in safe_headers:
            safe_headers["Authorization"] = "Bearer [REDACTED]"

        logger.debug("Request Details:")
        logger.debug(f"URL: {address}")
        logger.debug(f"Headers: {json.dumps(safe_headers, indent=2)}")
        logger.debug(f"Request Body: {json.dumps(body, indent=2)}")

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

        # Validate environment variables
        address = os.environ.get("OPENAI_API_BASE")
        key = os.environ.get("OPENAI_API_KEY")

        if not address:
            raise ValueError("OPENAI_API_BASE environment variable is not set")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        logger.debug(f"Using API base: {address}")

        # Prepare request
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            # Log request details before sending
            self._log_request_details(address, headers, body)

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
                    logger.error(f"Request failed with status {response.status_code}")
                    logger.error(f"Response content: {response.text}")
                    response.raise_for_status()

                for chunk in response.iter_lines(chunk_size=None):
                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    stem = "data: "
                    chunk = chunk[len(stem) :]

                    if chunk == b"[DONE]":
                        continue

                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse chunk: {chunk}")
                        logger.error(f"JSON decode error: {str(e)}")
                        continue

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        logger.error(f"API returned error: {error_msg}")
                        raise RuntimeError(error_msg)

                    tokens_received += 1
                    delta = data["choices"][0]["delta"]

                    if delta.get("content", None):
                        current_time = time.monotonic()
                        if not ttft:
                            ttft = current_time - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                current_time - most_recent_received_token_time
                            )
                        most_recent_received_token_time = current_time
                        generated_text += delta["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {str(e)}")
            if hasattr(e.response, "text"):
                logger.error(f"Response content: {e.response.text}")
            metrics[common_metrics.ERROR_MSG] = error_msg or str(e)
            metrics[common_metrics.ERROR_CODE] = error_response_code
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            metrics[common_metrics.ERROR_MSG] = error_msg or str(e)
            metrics[common_metrics.ERROR_CODE] = error_response_code

        # Calculate final metrics
        total_request_time = time.monotonic() - start_time
        output_throughput = (
            tokens_received / total_request_time if total_request_time > 0 else 0
        )

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
