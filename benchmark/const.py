import aiohttp

MILLISECONDS_TO_SECONDS_CONVERSION = 1000
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
WARMED = False


Q_STATIC = """SELECT
  case_id, arch, gpu, model, gpu_num, replicas, question_label, batch, 
  mean_input_tokens, mean_output_tokens, TTFT_P99, TPOT_P99, mean_TTFT, mean_TPOT, goodput_percentage,
  mean_prefill_throughput, mean_decode_throughput, mean_output_throughput, output_throughput, 
  mean_ITL, mean_E2EL,
  request_throughput,
  other_params, completed, backend{export_col}
FROM
  summary_data
WHERE
  case_id = '{case_id}'
"""

Q_DYNAMIC = """SELECT
  case_id, arch, gpu, model, gpu_num, replicas, question_label, batch, num_prompts,
  mean_input_tokens, mean_output_tokens, TTFT_P99, TPOT_P99, mean_TTFT, mean_TPOT, goodput_percentage,
  mean_prefill_throughput, mean_decode_throughput, mean_output_throughput, output_throughput, 
  mean_ITL, mean_E2EL, 
  request_throughput,
  other_params, completed, backend{export_col}
FROM
  summary_data
WHERE
  case_id = '{case_id}'
"""