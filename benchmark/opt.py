# from random import random
from collections import OrderedDict
import random
from typing import Dict, List
import asyncio

# from benchmark.util import logger


from .util import logger

async def find_optimal_batch(
    func,
    batch: List[int],
    func_kwargs: Dict = {},
    sparse_step: int = 20,
    dense_step: int = 10,
    condition: Dict[str, float] = {"ttft": 10000, "tpot": 100},
    strategy: str = "fast",
    result_key_map: dict = {"ttft": "TTFT_P99", "tpot": "TPOT_P99"}
):
    """
    寻找最优的batch参数
    
    参数:
        func: 评估函数，输入batch大小，输出包含吞吐量等指标的字典
        batch: 候选的batch大小列表
        sparse_step: 稀疏采样的步长，默认为50
        dense_step: 密集采样的步长，默认为20
        condition: 约束条件字典，键为指标名，值为最大允许值
        strategy: 搜索策略，"fast"为二分法快速搜索，"exhaustive"为穷举搜索
    
    返回:
        最佳batch值, 对应的结果字典, 算法执行步骤列表
    """
    # 参数验证
    if not batch:
        logger.error("错误：batch列表不能为空")
        return None, None
    if sparse_step <= 0 or dense_step <= 0:
        logger.error("错误：采样步长必须为正数")
        return None, None
    if strategy not in ["fast", "exhaustive"]:
        logger.error("错误：策略必须为'fast'或'exhaustive'")
        return None, None
    # 检查结果是否满足所有约束条件
    def check_conditions(result: Dict[str, float]) -> bool:
        for key, constraints in condition.items():
            if result_key_map[key] not in result:
                return False  # 如果结果中没有该指标，则不满足条件
            if result[result_key_map[key]] > constraints:
                return False
        return True
    # 存储所有满足约束条件的结果
    valid_results: Dict[int, Dict[str, float]] = {}
    total_results: Dict[int, Dict[str, float]] = {}
    total_evaluations = 0
    min_batch = min(batch)
    max_batch = max(batch)

    if strategy == "fast":
        left, right = min_batch, max_batch
        logger.info(f"1. 初始化搜索范围: [{left}, {right}]")
        # 二分搜索主循环
        while right - left > sparse_step:
            # 计算两个中间点
            mid = (left + right) // 2
            logger.info(f"  当前搜索范围: [{left}, {right}], 当前评估batch: {mid}")
            # 评估中间点
            if mid not in total_results:
                func_kwargs["max_concurrency"] = mid
                result = await func(**func_kwargs)
                total_results[mid] = result
                total_evaluations += 1
                if check_conditions(result):
                    valid_results[mid] = result
                    left = mid
                else:
                    right = mid
            else:
                index_result = total_results[mid]
                if check_conditions(index_result):
                    valid_results[mid] = index_result
                    left = mid
                else:
                    right = mid
        # 在最终范围内进行密集采样
        dense_samples = [b for b in range(left, right + 1, dense_step)]
        dense_samples = list(set(dense_samples) - set(valid_results.keys()))
        dense_samples.sort()
        logger.info(f"2. 在范围[{left}, {right}]内进行密集采样: {dense_samples}")
        
        for b in dense_samples:
            func_kwargs["max_concurrency"] = b
            result = await func(**func_kwargs)
            total_results[b] = result
            total_evaluations += 1
            if check_conditions(result):
                valid_results[b] = result

    else:
        best_batch = 0
        # 第一步：稀疏采样，了解大致趋势
        sparse_samples = [b for b in range(min_batch, max_batch + 1, sparse_step)]
        sparse_samples = list(set(sparse_samples))
        sparse_samples.sort()
        logger.info(f"1. 稀疏采样点: {sparse_samples} (共{len(sparse_samples)}个点)")
        
        for batch in sparse_samples:
            logger.info(f"  当前评估batch: {batch}, 当前最高batch：{best_batch}")
            func_kwargs["max_concurrency"] = batch
            result = await func(**func_kwargs)
            total_results[batch] = result
            total_evaluations += 1
            # 检查是否满足约束条件
            if check_conditions(result):
                valid_results[batch] = result
                if batch > best_batch:
                    best_batch = batch
    
        if not valid_results:
            logger.info(f"未找到满足约束条件的batch参数，共评估了{total_evaluations}个点")
            return total_results, valid_results
        # 第二步：分析稀疏采样结果，确定潜在最优区域, 进行密集采样
        dense_samples = [b for b in range(best_batch, best_batch + sparse_step + 1, dense_step)]
        dense_samples = list(set(dense_samples) - set(valid_results.keys())) 
        dense_samples.sort() # 排除已评估的点
        logger.info(f"2. 密集采样点: {dense_samples} (共{len(dense_samples)}个点)")
        
        for b in dense_samples:
            logger.info(f"  当前评估batch: {b}")
            func_kwargs["max_concurrency"] = b
            result = await func(**func_kwargs)
            total_results[b] = result
            total_evaluations += 1
            if check_conditions(result):
                valid_results[b] = result
    
    # 处理无有效结果的情况
    if not valid_results:
        logger.error(f"未找到满足约束条件的batch参数，共评估了{total_evaluations}个点")
        return total_results, valid_results

    # OrderedDict(sorted(valid_results.items()))
    return OrderedDict(sorted(total_results.items())), OrderedDict(sorted(valid_results.items()))


async def benchmark(max_concurrency, mode='dynamic'):
    # 假设吞吐量在batch=100左右达到最大
    await asyncio.sleep(0.5)
    base_throughput = 10 + 5 * (1 - (max_concurrency - 100)**2 / 5000)
    # 假设ttft随batch增大而增大
    base_ttft = 5000 + max_concurrency * 20
    base_tpot = 50 + max_concurrency * 20
    
    # 添加一些随机波动
    throughput = max(5, base_throughput + random.uniform(-1, 1))
    ttft = base_ttft + random.uniform(-500, 500)
    tpot = base_tpot + random.uniform(-15, 15)
    
    result = OrderedDict({"TTFT_P99": ttft, "throughput": throughput, "TPOT_P99": tpot})
    
    return result
    

async def main():
    random.seed(42)
    
    # 生成候选batch列表（从10到200，步长为5）
    candidate_batches = [10, 200]
    print(f"候选batch列表: 从{min(candidate_batches)}到{max(candidate_batches)}，共{len(candidate_batches)}个值\n")
    
    # 1. 使用快速搜索策略
    print("===== 使用快速搜索策略 =====")
    results = await find_optimal_batch(
        func=benchmark,
        batch=candidate_batches,
        sparse_step=20,
        dense_step=10,
        condition={"ttft": 10000, "tpot": 1000},  # 约束条件：ttft不超过8000，tpot不超过200
        strategy="fast"
    )
    print(results[0])
    print(results[1])
    
    # 2. 使用穷举搜索策略进行对比
    print("\n\n===== 使用穷举搜索策略 =====")
    results = await find_optimal_batch(
        func=benchmark,
        batch=candidate_batches,
        sparse_step=20,
        dense_step=10,
        condition={"ttft": 10000, "tpot": 1000},
        strategy="exhaustive"
    )
    print(results[0])
    print(results[1])

if __name__ == "__main__":
    # asyncio.run(main())
    # asyncio.run(benchmark(200))
    asyncio.run(main())
