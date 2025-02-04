from dataclasses import dataclass
import time
import random
import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from transformers import PreTrainedTokenizerFast
from llmperf.common import construct_clients
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf import common_metrics


@dataclass
class BenchmarkConfig:
    model: str
    llm_api: str = "openai"
    test_timeout_s: int = 90
    max_num_completed_requests: int = 500
    num_concurrent_requests: int = 1
    max_output_tokens: int = 100
    additional_sampling_params: Optional[Dict[str, Any]] = None
    use_repeated_prompt: bool = False

    def __post_init__(self):
        print(f"Debug: Initializing BenchmarkConfig with model={self.model}, llm_api={self.llm_api}")
        self.additional_sampling_params = self.additional_sampling_params or {}
        print(f"Debug: Set additional_sampling_params={self.additional_sampling_params}")


class TokenMetricsCalculator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        print("Debug: Initializing TokenMetricsCalculator")
        self.tokenizer = tokenizer

    def get_token_length(self, text: str) -> int:
        print(f"Debug: Getting token length for text of length {len(text)}")
        try:
            tokens = self.tokenizer.encode(text)
            print(f"Debug: Tokenization successful, token count: {len(tokens)}")
            return len(tokens)
        except Exception as e:
            print(f"Debug: Tokenization failed with error: {str(e)}")
            raise

    def calculate_request_metrics(
        self, request_metrics: Dict[str, Any], generated_text: str
    ) -> Dict[str, Any]:
        print("Debug: Calculating request metrics")
        num_output_tokens = self.get_token_length(generated_text)
        print(f"Debug: Output tokens: {num_output_tokens}")
        metrics = request_metrics.copy()

        # Calculate per-token metrics
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
        metrics[common_metrics.NUM_TOTAL_TOKENS] = metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
        print(f"Debug: Total tokens: {metrics[common_metrics.NUM_TOTAL_TOKENS]}")

        if num_output_tokens:
            metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
        else:
            metrics[common_metrics.INTER_TOKEN_LAT] = 0
        print(f"Debug: Inter-token latency: {metrics[common_metrics.INTER_TOKEN_LAT]}")

        # Calculate throughput
        generation_time = metrics[common_metrics.E2E_LAT] - metrics[common_metrics.TTFT]
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
            num_output_tokens / generation_time if generation_time > 0 else 0
        )
        print(f"Debug: Output throughput: {metrics[common_metrics.REQ_OUTPUT_THROUGHPUT]}")

        return metrics


class PerformanceAnalyzer:
    @staticmethod
    def calculate_summary_statistics(series: pd.Series) -> Dict[str, Any]:
        print(f"Debug: Calculating summary statistics for series of length {len(series)}")
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        stats = {
            "quantiles": {f"p{int(q * 100)}": v for q, v in quantiles.items()},
            "mean": series.mean(),
            "min": series.min(),
            "max": series.max(),
            "stddev": series.std(),
        }
        print(f"Debug: Summary statistics: {stats}")
        return stats

    @staticmethod
    def analyze_metrics(
        metrics: List[Dict[str, Any]], start_time: float, end_time: float
    ) -> Dict[str, Any]:
        print(f"Debug: Analyzing metrics for {len(metrics)} requests")
        df = pd.DataFrame(metrics)
        df_success = df[df[common_metrics.ERROR_CODE].isna()]
        print(f"Debug: Found {len(df_success)} successful requests")

        results = {}

        # Calculate statistics for key metrics
        metric_keys = [
            common_metrics.INTER_TOKEN_LAT,
            common_metrics.TTFT,
            common_metrics.E2E_LAT,
            common_metrics.REQ_OUTPUT_THROUGHPUT,
            common_metrics.NUM_INPUT_TOKENS,
            common_metrics.NUM_OUTPUT_TOKENS,
        ]

        for key in metric_keys:
            print(f"Debug: Analyzing metric: {key}")
            series = pd.Series(df_success[key].dropna())
            results[key] = PerformanceAnalyzer.calculate_summary_statistics(series)

        # Error analysis
        error_codes = df[common_metrics.ERROR_CODE].dropna()
        num_errors = len(error_codes)
        print(f"Debug: Found {num_errors} errors")
        results.update(
            {
                common_metrics.NUM_REQ_STARTED: len(metrics),
                common_metrics.NUM_ERRORS: num_errors,
                common_metrics.ERROR_RATE: num_errors / len(metrics) if metrics else 0,
                common_metrics.ERROR_CODE_FREQ: (
                    dict(error_codes.value_counts()) if num_errors else {}
                ),
            }
        )

        # Overall throughput calculations
        mean_ttft = df_success[common_metrics.TTFT].mean()
        generation_time = end_time - start_time - mean_ttft
        total_output_tokens = df_success[common_metrics.NUM_OUTPUT_TOKENS].sum()
        print(f"Debug: Total output tokens: {total_output_tokens}")
        print(f"Debug: Total generation time: {generation_time}")

        results.update(
            {
                common_metrics.OUTPUT_THROUGHPUT: (
                    total_output_tokens / generation_time if generation_time > 0 else 0
                ),
                common_metrics.NUM_COMPLETED_REQUESTS: len(df_success),
                common_metrics.COMPLETED_REQUESTS_PER_MIN: len(df_success)
                / (end_time - start_time)
                * 60,
            }
        )

        return results


class LLMBenchmark:
    def __init__(self, config: BenchmarkConfig, tokenizer: PreTrainedTokenizerFast):
        print("Debug: Initializing LLMBenchmark")
        self.config = config
        self.metrics_calculator = TokenMetricsCalculator(tokenizer)
        self.analyzer = PerformanceAnalyzer()
        random.seed(11111)
        self.fixed_message = None

    def load_messages(self, jsonl_path: str) -> List[Dict[str, Any]]:
        print(f"Debug: Loading messages from {jsonl_path}")
        with open(jsonl_path) as f:
            messages = [json.loads(line) for line in f]
        print(f"Debug: Loaded {len(messages)} messages")
        return messages

    def run_benchmark(
        self, jsonl_path: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        print("Debug: Starting benchmark run")
        messages = self.load_messages(jsonl_path)
        if self.config.use_repeated_prompt:
            self.fixed_message = random.choice(messages)
            print("Debug: Using fixed message for repeated prompts")

        clients = construct_clients(
            llm_api=self.config.llm_api, num_clients=self.config.num_concurrent_requests
        )
        print(f"Debug: Created {len(clients)} clients")

        req_launcher = RequestsLauncher(clients)
        completed_requests = []

        start_time = time.monotonic()
        print(f"Debug: Benchmark started at {start_time}")

        while (
            time.monotonic() - start_time < self.config.test_timeout_s
            and len(completed_requests) < self.config.max_num_completed_requests
        ):
            print("Debug: Processing new batch of requests")
            completed_requests.extend(
                self._process_request_batch(req_launcher, messages)
            )
            print(f"Debug: Total completed requests: {len(completed_requests)}")

        # Process any remaining requests
        print("Debug: Processing remaining requests")
        completed_requests.extend(self._process_request_batch(req_launcher, messages))

        end_time = time.monotonic()
        print(f"Debug: Benchmark ended at {end_time}")

        # Generate summary
        print("Debug: Generating summary")
        summary = self.analyzer.analyze_metrics(
            completed_requests, start_time, end_time
        )
        metadata = {
            "model": self.config.model,
            "num_concurrent_requests": self.config.num_concurrent_requests,
            "additional_sampling_params": self.config.additional_sampling_params,
            "use_repeated_prompt": self.config.use_repeated_prompt,
            "results": summary,
        }
        print(f"Debug: Benchmark complete with {len(completed_requests)} total requests")

        return metadata, completed_requests

    def _process_request_batch(
        self, req_launcher: RequestsLauncher, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        print("Debug: Processing request batch")
        message = self.fixed_message if self.fixed_message else random.choice(messages)
        system_prompt = message["messages"][0]["content"]
        user_prompt = message["messages"][1]["content"]
        print(f"Debug: System prompt length: {len(system_prompt)}")
        print(f"Debug: User prompt length: {len(user_prompt)}")
        
        num_input_tokens = self.metrics_calculator.get_token_length(
            system_prompt + user_prompt
        )
        print(f"Debug: Input tokens: {num_input_tokens}")

        sampling_params = {"max_tokens": self.config.max_output_tokens}
        if self.config.additional_sampling_params:
            sampling_params.update(self.config.additional_sampling_params)
        print(f"Debug: Using sampling parameters: {sampling_params}")

        request_config = RequestConfig(
            model=self.config.model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            num_input_tokens=num_input_tokens,
            sampling_params=sampling_params,
            llm_api=self.config.llm_api,
        )

        req_launcher.launch_requests(request_config)
        processed_metrics = []

        print("Debug: Processing responses")
        for metrics, gen_text, _ in req_launcher.get_next_ready():
            print(f"Debug: Processing response with text length: {len(gen_text)}")
            processed_metrics.append(
                self.metrics_calculator.calculate_request_metrics(metrics, gen_text)
            )

        print(f"Debug: Processed {len(processed_metrics)} responses")
        return processed_metrics


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    max_output_tokens: int,
    tokenizer: PreTrainedTokenizerFast,
    jsonl_path: str,
    use_repeated_prompt: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run a token-based benchmark for an LLM model."""
    print("Debug: Starting token benchmark")
    config = BenchmarkConfig(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        num_concurrent_requests=num_concurrent_requests,
        max_output_tokens=max_output_tokens,
        use_repeated_prompt=use_repeated_prompt,
    )
    print(f"Debug: Created benchmark config for model {model}")

    benchmark = LLMBenchmark(config, tokenizer)
    print("Debug: Created benchmark instance")
    return benchmark.run_benchmark(jsonl_path)
