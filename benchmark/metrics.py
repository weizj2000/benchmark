# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .const import MILLISECONDS_TO_SECONDS_CONVERSION
from .models import RequestFuncOutput, BenchmarkMetrics
from .util import logger


class MetricsCalculator:
    """性能指标计算类"""

    @staticmethod
    def calculate_metrics(outputs: Tuple[RequestFuncOutput],
                          dur_s: float,
                          selected_percentiles: List[float],
                          goodput_config_dict: Dict[str, float],
                          ):
        actual_output_lens: List[int] = []
        total_input = 0
        completed = 0
        good_completed = 0
        itls: List[float] = []
        tpots: List[float] = []
        all_tpots: List[float] = []
        ttfts: List[float] = []
        e2els: List[float] = []
        out_throughputs: List[float] = []  # throughput of every request
        prefill_throughputs: List[float] = []  # include ttft and time of queue
        decode_throughputs: List[float] = []

        for i in range(len(outputs)):
            if outputs[i].success:
                logger.debug(outputs[i])
                output_len = outputs[i].completion_len
                actual_output_lens.append(output_len)
                total_input += outputs[i].prompt_len

                tpot, throughput = 0, 0
                if output_len > 1:
                    tpot = (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                    tpots.append(tpot)

                    throughput = output_len / outputs[i].latency
                    out_throughputs.append(throughput)

                    prefill_throughput = outputs[i].prompt_len / outputs[i].ttft
                    prefill_throughputs.append(prefill_throughput)

                    decode_throughput = (output_len - 1) / (outputs[i].latency - outputs[i].ttft)
                    decode_throughputs.append(decode_throughput)

                # Note: if output_len <= 1, we regard tpot as 0 for goodput
                all_tpots.append(tpot)
                itls += outputs[i].itl
                ttfts.append(outputs[i].ttft)
                e2els.append(outputs[i].latency)
                completed += 1
            else:
                actual_output_lens.append(0)

        mean_goodput_ttft = -1
        mean_goodput_tpot = -1
        mean_goodput_e2el = -1
        mean_goodput_throughput = -1
        if goodput_config_dict:
            valid_metrics, slo_values = {}, {}
            valid_metrics['ttft'] = ttfts
            valid_metrics['tpot'] = all_tpots
            valid_metrics['e2el'] = e2els
            valid_metrics['throughput'] = decode_throughputs  # the same as the above form

            if "ttft" in goodput_config_dict:
                slo_values['ttft'] = goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            else:
                slo_values['ttft'] = float('inf')

            if "tpot" in goodput_config_dict:
                slo_values['tpot'] = goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            else:
                slo_values['tpot'] = float('inf')

            if "e2el" in goodput_config_dict:
                slo_values['e2el'] = goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            else:
                slo_values['e2el'] = float('inf')

            if 'throughput' in goodput_config_dict:
                slo_values['throughput'] = goodput_config_dict["throughput"]
            else:
                slo_values['throughput'] = -float('inf')

            good_ttfts, good_tpots, good_e2els, good_throughputs = np.array([]), np.array([]), np.array([]), np.array(
                [])
            for _ttft, _tpot, _e2el, _throughput in zip(valid_metrics['ttft'], valid_metrics['tpot'],
                                                        valid_metrics['tpot'], valid_metrics['throughput']):
                if (_ttft <= slo_values['ttft'] and _tpot <= slo_values['tpot'] and
                        _e2el <= slo_values['e2el'] and _throughput >= slo_values['throughput']):
                    good_ttfts = np.append(good_ttfts, _ttft)
                    good_tpots = np.append(good_tpots, _tpot)
                    good_e2els = np.append(good_e2els, _e2el)
                    good_throughputs = np.append(good_throughputs, _throughput)

            assert len(good_ttfts) == len(good_tpots) == len(good_e2els) == len(good_throughputs)
            logger.debug(f"valid_metrics: {valid_metrics}")

            good_completed = len(good_ttfts)

            if good_completed > 0:
                mean_goodput_ttft = np.mean(good_ttfts)
                mean_goodput_tpot = np.mean(good_tpots)
                mean_goodput_e2el = np.mean(good_e2els)
                mean_goodput_throughput = np.mean(good_throughputs)
            else:
                mean_goodput_ttft, mean_goodput_tpot, mean_goodput_e2el, mean_goodput_throughput = np.nan, np.nan, np.nan, np.nan

        if completed == 0:
            logger.warning(
                "!!!All requests failed. This is likely due to a misconfiguration "
                "on the benchmark arguments.")
            completed = float('-inf')

        metrics = BenchmarkMetrics(
            completed=completed,
            mean_input=int(total_input / completed),
            mean_output=int(np.mean(actual_output_lens)),
            total_input=total_input,
            total_output=sum(actual_output_lens),
            request_throughput=completed / dur_s,
            request_goodput=good_completed / dur_s,
            goodput_percentage=(good_completed / completed) * 100,

            output_throughput=sum(actual_output_lens) / dur_s,
            mean_output_throughput=float(np.mean(out_throughputs)),
            total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,

            mean_goodput_ttft=mean_goodput_ttft,
            mean_goodput_tpot=mean_goodput_tpot,
            mean_goodput_e2el=mean_goodput_e2el,
            mean_goodput_throughput=mean_goodput_throughput,

            mean_ttft_ms=np.mean(ttfts or 0) *
                         1000,  # ttfts is empty if streaming is not supported by backend
            std_ttft_ms=np.std(ttfts or 0) * 1000,
            median_ttft_ms=np.median(ttfts or 0) * 1000,
            percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                                 for p in selected_percentiles],
            mean_tpot_ms=np.mean(tpots or 0) * 1000,
            std_tpot_ms=np.std(tpots or 0) * 1000,
            median_tpot_ms=np.median(tpots or 0) * 1000,
            percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                                 for p in selected_percentiles],
            mean_itl_ms=np.mean(itls or 0) * 1000,
            std_itl_ms=np.std(itls or 0) * 1000,
            median_itl_ms=np.median(itls or 0) * 1000,
            percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                                for p in selected_percentiles],
            mean_e2el_ms=np.mean(e2els or 0) * 1000,
            std_e2el_ms=np.std(e2els or 0) * 1000,
            median_e2el_ms=np.median(e2els or 0) * 1000,
            percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                                 for p in selected_percentiles],

            mean_prefilling_throughput=np.mean(prefill_throughputs or 0),
            std_prefilling_throughput=np.std(prefill_throughputs or 0),
            median_prefilling_throughput=np.median(prefill_throughputs or 0),
            percentiles_prefilling_throughput=[(p, np.percentile(prefill_throughputs or 0, p))
                                               for p in selected_percentiles],

            mean_decoding_throughput=np.mean(decode_throughputs or 0),
            std_decoding_throughput=np.std(decode_throughputs or 0),
            median_decoding_throughput=np.median(decode_throughputs or 0),
            percentiles_decoding_throughput=[(p, np.percentile(decode_throughputs or 0, p))
                                             for p in selected_percentiles],
        )

        return metrics, actual_output_lens
