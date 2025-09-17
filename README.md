# benchmark
Benchmark for LLM


## 示例

### Static Benchmark
```shell
# static
benchmark --model /data/model/Qwen3-8B \
--base-url http://localhost:30007 \
--result-dir /Users/weizhanjun/Workspace/PythonProjects/benchmark/temp \
--dataset-path /Users/weizhanjun/Workspace/PythonProjects/benchmark/benchmark/dataset/filtered.json \
--dataset-name filtered \
--max-concurrency 1,2 \
--input-len 1024 \
--output-len 1024 \
--metadata arch=x86 gpu="NVIDIA A30" gpu_num=1 \
  replicas=1 backend=sgalng410 other_params="test"

```

### Dynamic Benchmark
```shell
# dynamic
benchmark --model /data/model/Qwen3-8B \
--base-url http://localhost:30007 \
--result-dir /Users/weizhanjun/Workspace/PythonProjects/benchmark/temp \
--dataset-path /Users/weizhanjun/Workspace/PythonProjects/benchmark/benchmark/dataset/ShareGPT_V3_10000.json \
--dataset-name sharegpt \
--num-prompts 100 \
--max-concurrency 20,50 \
--tokenizer-path /Users/weizhanjun/Workspace/PythonProjects/benchmark/benchmark/dataset/tokenizer \
--metadata arch=x86 gpu="NVIDIA A30" gpu_num=1 \
  replicas=1 backend=sgalng410 other_params="test"

```
