import random
import json
from typing import List
import tqdm
import os
from multiprocessing import Pool
import multiprocessing
from ..tokenizor import prepare_and_save_data, dpTokenizer, ts
# ---------------------------------------------------------------------------
# DP任务的核心函数
# ---------------------------------------------------------------------------

def generate_triangle(height: int) -> List[List[int]]:
    """生成一个指定高度的数字三角形，填充值为0或1。"""
    # 为了让路径和更有区分度，可以使用更大的随机数，但0/1最符合词表最小化需求。
    return [[random.randint(0, 1) for _ in range(i + 1)] for i in range(height)]

def triangle_to_string(triangle: List[List[int]]) -> str:
    """将三角形数据结构转换为多行字符串。"""
    return "\n".join(" ".join(map(str, row)) for row in triangle)

def solve_dp(triangle: List[List[int]]) -> int:
    """一次性计算出DP问题的最终解，不记录中间步骤。"""
    if not triangle:
        return 0
    dp = list(triangle[-1])
    for i in range(len(triangle) - 2, -1, -1):
        next_dp = []
        for j in range(len(triangle[i])):
            next_dp.append(triangle[i][j] + max(dp[j], dp[j+1]))
        dp = next_dp
    return dp[0]

def generate_cot_steps_dp(triangle: List[List[int]], t: int) -> List[dict]:
    """
    生成DP求解过程的CoT（Chain-of-Thought）步骤。
    参数 t 控制每个步骤向上推理的层数。
    """
    if t < 1:
        t = 1 # 保证t至少为1
        
    steps = []
    # DP的初始状态是三角形的最后一行
    current_dp = list(triangle[-1])
    
    # i 是当前处理的三角形层的索引，从倒数第二层开始
    i = len(triangle) - 2
    while i >= 0:
        # 记录当前步骤开始前的DP状态
        expression_before = " ".join(map(str, current_dp))
        
        # 确定此步骤向上计算的终点层
        end_layer_idx = max(i - t + 1, 0)
        
        # 在一个CoT步骤中，执行t层DP计算
        temp_dp = list(current_dp)
        for layer_idx in range(i, end_layer_idx - 1, -1):
            next_dp = []
            row_values = triangle[layer_idx]
            for j in range(len(row_values)):
                new_val = row_values[j] + max(temp_dp[j], temp_dp[j+1])
                next_dp.append(new_val)
            temp_dp = next_dp
        
        current_dp = temp_dp
        expression_after = " ".join(map(str, current_dp))
        
        # 组装CoT步骤字典
        step = {
            # "sub_expression" 字段可以省略，因为before/after已经足够清晰
            # 这里保留它是为了和您的旧结构完全对应
            "sub_expression": f"Computing from layer {i+1} up to {end_layer_idx+1}",
            "value": expression_after,
            "expression_before": expression_before,
            "expression_after": expression_after
        }
        steps.append(step)
        
        # 更新i到下一个要处理的层
        i = end_layer_idx - 1
        
    return steps


def generate_data(num_samples: int, T: int, t: int, mix: bool) -> List[dict]:
    """生成一批DP任务数据样本。"""
    data = []
    for _ in tqdm.tqdm(range(num_samples), desc=f"Generating data for t={t}"):
        # 根据mix参数决定是使用固定高度还是随机高度
        # random_T = random.randint(12, T) if mix else T
        random_T = random.sample(list(range(t, T + 1)), 1)[0] if mix else T
        
        triangle = generate_triangle(random_T)
        question = triangle_to_string(triangle)
        answer = solve_dp(triangle)
        
        # 生成CoT步骤
        cot_steps = generate_cot_steps_dp(triangle, t)
        
        data.append({
            "depth": random_T,
            "ops_per_step": t, 
            "question": question,
            "answer": str(answer),
            "cot": cot_steps
        })
    
    return data

def write_jsonl(data: List[dict], filename: str, write_type='w'):
    """将数据以jsonl格式写入文件。"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, write_type, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



if __name__ == "__main__":
    T = 40  # max depth of tree
    t = 8  # one step cot length
    test_samples_each = 100

    
    for temp_T in range(t,T+1):
        data = generate_data(test_samples_each, temp_T, t, False)
        write_jsonl(data, f'synthetic/dataset/data/dp/test/{temp_T}.jsonl')