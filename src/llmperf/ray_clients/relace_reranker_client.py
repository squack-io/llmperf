import json
import os
import time
from typing import Any, Dict, List, Tuple

import ray
import requests

from llmperf.models import RankerConfig
from llmperf import common_metrics


def parse_json_list(json_str):
    """Convert a JSON string representation of a list to a Python list."""
    try:
        if isinstance(json_str, str):
            return json.loads(json_str)
        return json_str  # Return as is if already parsed
    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"Invalid JSON list format: {json_str}")


@ray.remote
class RelaceRerankerClient:
    """Client for Relace API."""

    def llm_request(
        self, ranker_config: RankerConfig
    ) -> Tuple[Dict[str, Any], str, RankerConfig]:
        query = ranker_config.query
        codebase_str = ranker_config.codebase
        codebase = parse_json_list(codebase_str)

        body = {"query": query, "codebase": codebase, "token_limit": 999999999}
        error_response_code = -1
        error_msg = ""
        total_request_time = 0
        generated_text = ""

        metrics = {}
        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        address = os.environ.get("RELACE_API_BASE")
        if not address:
            raise ValueError("the environment variable RELACE_API_BASE must be set.")
        key = os.environ.get("RELACE_API_KEY")
        if not key:
            raise ValueError("the environment variable RELACE_API_KEY must be set.")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += "v2/code/rank"

        try:
            with requests.post(
                address,
                json=body,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()

                data = response.json()
                if "error" in data:
                    error_msg = data["error"]["message"]
                    error_response_code = data["error"]["code"]
                    raise RuntimeError(data["error"]["message"])

                generated_text = json.dumps(data)

            total_request_time = time.monotonic() - start_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        # Set metrics for non-streaming request
        metrics[common_metrics.E2E_LAT] = total_request_time

        return metrics, generated_text, ranker_config
