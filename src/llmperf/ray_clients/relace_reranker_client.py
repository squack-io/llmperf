import json
import os
import time
from typing import Any, Dict, List, Tuple

import ray
import requests

from llmperf.models import RankerConfig
from llmperf import common_metrics

import ast


@ray.remote
class RelaceRerankerClient:
    """Client for Relace API."""

    def llm_request(
        self, ranker_config: RankerConfig
    ) -> Tuple[Dict[str, Any], str, RankerConfig]:
        query = ranker_config.query
        codebase_str = ranker_config.codebase
        codebase = ast.literal_eval(codebase_str)

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

        print(f"Request URL: {address}")
        print(
            f"Request headers: {{'Authorization': 'Bearer <REDACTED>', 'Content-Type': '{headers['Content-Type']}'}}"
        )
        print(
            f"Request body structure: {json.dumps({k: '...' if k == 'codebase' else body[k] for k in body})}"
        )
        print(f"Codebase length: {len(json.dumps(codebase))} characters")
        print(f"Query: {query}")

        try:
            print("Sending request to Relace API...")
            with requests.post(
                address,
                json=body,
                headers=headers,
            ) as response:
                print(f"Response status code: {response.status_code}")
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    print(f"Error response text: {error_msg}")
                    response.raise_for_status()

                data = response.json()
                print(f"Response data structure: {list(data.keys())}")
                if "error" in data:
                    error_msg = data["error"]["message"]
                    error_response_code = data["error"]["code"]
                    print(
                        f"Error in response data: {error_msg}, code: {error_response_code}"
                    )
                    raise RuntimeError(data["error"]["message"])

                generated_text = json.dumps(data)

            total_request_time = time.monotonic() - start_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")
            print(f"Error message: {error_msg}")
            print(f"Error code: {error_response_code}")

        # Set metrics for non-streaming request
        metrics[common_metrics.E2E_LAT] = total_request_time
        print(f"Total request time: {total_request_time} seconds")

        return metrics, generated_text, ranker_config
