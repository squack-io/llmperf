@ray.remote
class RelaceClient:
    """Client for Relace API."""

    def llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[Dict[str, Any], str, RequestConfig]:
        print("Debug: Starting llm_request in RelaceClient")
        print(
            f"Debug: RequestConfig num_input_tokens: {request_config.num_input_tokens}"
        )

        user_prompt = request_config.user_prompt
        system_prompt = request_config.system_prompt
        print(
            f"Debug: Prompt lengths - system: {len(system_prompt)}, user: {len(user_prompt)}"
        )

        message = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        print(f"Debug: Request body prepared with model: {model}")

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}
        print("Debug: Initialized empty metrics dictionary")

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        # API setup debug
        address = os.environ.get("OPENAI_API_BASE")
        print(f"Debug: Using API address: {address}")

        key = os.environ.get("OPENAI_API_KEY")
        print("Debug: API key present:", bool(key))

        headers = {"Authorization": f"Bearer {key}"}
        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"
        print(f"Debug: Final API endpoint: {address}")

        try:
            print("Debug: Starting API request")
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=180,
                headers=headers,
            ) as response:
                print(f"Debug: Initial response status: {response.status_code}")
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    print(
                        f"Debug: Error response received: {error_response_code} - {error_msg}"
                    )
                    response.raise_for_status()

                print("Debug: Processing response stream")
                for chunk in response.iter_lines(chunk_size=None):
                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    stem = "data: "
                    chunk = chunk[len(stem) :]
                    if chunk == b"[DONE]":
                        continue

                    tokens_received += 1
                    data = json.loads(chunk)

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        print(f"Debug: Error in response data: {error_msg}")
                        raise RuntimeError(data["error"]["message"])

                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                            print(f"Debug: First token received. TTFT: {ttft}")
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += delta["content"]

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time
            print(
                f"Debug: Request completed. Total time: {total_request_time}, Tokens received: {tokens_received}"
            )

        except Exception as e:
            print(f"Debug: Exception in API request: {str(e)}")
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        print("Debug: Building final metrics")
        print(f"Debug: Inter-token latency total: {sum(time_to_next_token)}")
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)

        print(f"Debug: Setting TTFT: {ttft}")
        metrics[common_metrics.TTFT] = ttft

        print(f"Debug: Setting E2E latency: {total_request_time}")
        metrics[common_metrics.E2E_LAT] = total_request_time

        print(f"Debug: Setting output throughput: {output_throughput}")
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput

        print(
            f"Debug: Setting total tokens. Received: {tokens_received}, Input: {request_config.num_input_tokens}"
        )
        metrics[common_metrics.NUM_TOTAL_TOKENS] = (
            tokens_received + request_config.num_input_tokens
        )

        print(f"Debug: Setting output tokens: {tokens_received}")
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received

        print(f"Debug: Setting input tokens: {request_config.num_input_tokens}")
        metrics[common_metrics.NUM_INPUT_TOKENS] = request_config.num_input_tokens

        print("Debug: Final metrics structure:")
        for key, value in metrics.items():
            print(f"Debug: {key}: {value}")

        return metrics, generated_text, request_config
