# benchmark
Benchmark for LLM

## 打包
在项目根目录下，执行`python setup.py bdist_wheel`，生成whl包；

## 安装
执行安装 `pip install -e .`

## 使用示例

### Static Benchmark
```shell
# static
benchmark --model /data/Qwen3-235B-A22B-Instruct-2507-FP8 \
--base-url http://127.0.0.1:30007 \
--result-dir /workspace/result \
--dataset-path /workspace/dataset/filtered.json \
--dataset-name filtered \
--max-concurrency 1,2 \
--input-len 1024 \
--output-len 1024 \
--metadata arch=x86 gpu="NVIDIA H20" gpu_num=4 \
  replicas=1 backend=sgalng410 other_params="default"
```

### Dynamic Benchmark
```shell
# dynamic
benchmark --model /data/Qwen3-235B-A22B-Instruct-2507-FP8 \
--base-url http://127.0.0.1:30007 \
--result-dir /workspace/result \
--dataset-path /workspace/dataset/ShareGPT_V3_10000.json \
--dataset-name sharegpt \
--num-prompts 100 \
--max-concurrency 20,50 \
--tokenizer-path /workspace/dataset/tokenizer \
--dynamic-input-len 1024 \
--dynamic-output-len 1024 \
--dynamic-prompt-len-scale 0.1 \
--enable-same-prompt \
--metadata arch=x86 gpu="NVIDIA H20" gpu_num=4 \
  replicas=1 backend=sgalng410 other_params="default"
```


### Dynamic Benchmark Auto Find Batch
```shell
# dynamic
benchmark --model /data/Qwen3-235B-A22B-Instruct-2507-FP8 \
--base-url http://127.0.0.1:30007 \
--result-dir /workspace/result \
--dataset-path /workspace/dataset/ShareGPT_V3_10000.json \
--dataset-name sharegpt \
--num-prompts 200 \
--max-concurrency 20,150 \
--tokenizer-path /workspace/dataset/tokenizer \
--dynamic-input-len 1024 \
--dynamic-output-len 1024 \
--dynamic-prompt-len-scale 0.1 \
--enable-same-prompt \
--enable-auto-batch \
--sparse-step 50 \
--dense-step 20 \
--dynamic-strategy "fast" \
--dynamic-result-key-map '{"ttft": "TTFT_P99", "tpot": "TPOT_P99"}' \
--metadata arch=x86 gpu="NVIDIA H20" gpu_num=4 \
  replicas=1 backend=sgalng410 other_params="default"

# 其中，dynamic-result-key-map 是评估标准键值映射
# 默认 {"ttft": "TTFT_P99", "tpot": "TPOT_P99"} 取值P99
# {"ttft": "mean_TTFT", "tpot": "mean_TPOT"} 取值平均值
```

### Docker Run Benchmark
```shell
docker run -it --rm \
--network=host \
--ipc=host \
--privileged=true \
-v /home/zjwei/result:/workspace/result \
sangfor.com/benchmark:v0.0.7 \
benchmark --model /data/Qwen3-235B-A22B-Instruct-2507-FP8 \
--base-url http://127.0.0.1:30007 \
--result-dir /workspace/result \
--result-dirname result \
--dataset-path /workspace/dataset/ShareGPT_V3_10000.json \
--dataset-name sharegpt \
--num-prompts 200 \
--max-concurrency 20,200 \
--tokenizer-path /workspace/dataset/tokenizer \
--dynamic-input-len 1024 \
--dynamic-output-len 1024 \
--dynamic-prompt-len-scale 0.1 \
--enable-same-prompt \
--enable-auto-batch \
--sparse-step 50 \
--dense-step 20 \
--dynamic-strategy "fast" \
--dynamic-result-key-map '{"ttft": "TTFT_P99", "tpot": "TPOT_P99"}' \
--metadata arch=x86 gpu="NVIDIA H20" gpu_num=4 \
  replicas=1 backend=sgalng410 other_params="default"
```