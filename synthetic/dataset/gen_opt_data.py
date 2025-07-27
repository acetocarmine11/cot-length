import random
import json
from typing import List, Union
import tqdm
import os
import sys
sys.path.append('..')
from tokenizor import prepare_and_save_data, arithmeticTokenizer, ff_mod, ts
from multiprocessing import Pool
import pandas as pd
import numpy as np

# 定义解析文件的函数
def parse_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # 跳过空行
                parts = line.split(', ')
                model_size = int(parts[0].split('=')[1])
                test_T = int(parts[1].split('=')[1])
                t = int(parts[2].split('=')[1])
                accuracy = float(parts[3].split(': ')[1])
                # if test_T in [8, 72,80]:
                #     continue
                # if test_T not in [12,24,36,48,60]:
                    # continue
                # if test_T not in [12,20,28,36,42,50,58,66]:
                    # continue
                data.append([model_size, test_T, t, accuracy])
    return pd.DataFrame(data, columns=['model_size', 'test_T', 't', 'accuracy'])

# 加载数据
caption = 'alot'
filename = f'eval_results/eval_results_{caption}.txt'
data = parse_txt_file(filename)

# 找到每组 (model_size, test_T) 下最大 accuracy 所对应的最小 t
heatmap_data = data.groupby(['model_size', 'test_T']).apply(
    lambda group: group[group['accuracy'] == group['accuracy'].max()].sort_values('t',ascending=False).iloc[0]
).reset_index(drop=True)

# 创建一个嵌套字典
opt_map = {}

# 遍历每一组数据
for _, row in heatmap_data.iterrows():
    model_size = row['model_size']
    test_T = row['test_T']
    min_t = row['t']
    
    # 如果 model_size 还没有在字典中，初始化为空字典
    if model_size not in opt_map:
        opt_map[model_size] = {}
    
    # 将 min_t 填入对应的 test_T 下
    opt_map[model_size][test_T] = min_t



class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.left = None
        self.right = None

    @property
    def is_leaf(self):
        return not self.left and not self.right

def generate_tree(depth: int, ff_mod: int, parent=None) -> Node:
    if depth == 0:
        return Node(random.randint(0, ff_mod-1), parent)
    
    # root = Node(random.choice(['+', '*']), parent)
    root = Node(random.choice(['+']), parent)
    if random.random() < 0.5:
        root.left = Node(random.randint(0, ff_mod-1), root)
        root.right = generate_tree(depth-1, ff_mod, root)
    else:
        root.right = Node(random.randint(0, ff_mod-1), root)
        root.left = generate_tree(depth-1, ff_mod, root)
    return root



def tree_to_expression(node: Node) -> str:
    if isinstance(node.value, int):
        return str(node.value)
    
    left = tree_to_expression(node.left)
    right = tree_to_expression(node.right)
    
    if node.value == '+':
        return f"+ {left} {right}"
    else:  # '*'
        return f"*{left} {right}"

def ff_evaluate(node: Node, ff_mod: int) -> int:
    if isinstance(node.value, int):
        return node.value % ff_mod
    
    left = ff_evaluate(node.left, ff_mod)
    right = ff_evaluate(node.right, ff_mod)
    
    if node.value == '+':
        return (left + right) % ff_mod
    else:  # '*'
        return (left * right) % ff_mod

def replace_node_in_parent(target, new_node):
    if target.parent:
        if target.parent.left == target:
            target.parent.left = new_node
        else:
            target.parent.right = new_node

def generate_cot_steps(node: Node, ff_mod: int, t: int) -> List[dict]:
    steps = []
    
    def find_and_update(current_node: Node):

        def process_one_step(cur_node: Node):
            value = ff_evaluate(cur_node, ff_mod)
            expression = tree_to_expression(cur_node)
            step = {
                "sub_expression": expression,
                "value": value,
                "expression_before": tree_to_expression(node),
                "expression_after": None  # 将在更新树后填充
            }
            new_node = Node(value, parent = cur_node.parent)
            #还需要修改parent的指针索引
            replace_node_in_parent(target = cur_node, new_node = new_node)
            step["expression_after"] = tree_to_expression(node)
            steps.append(step)

            
            
        
        cur_t = 1
        while current_node is not None:
            if cur_t >= t:
                cur_t = 1
                process_one_step(cur_node = current_node)
                current_node = current_node.parent
                cur_t += 1
                continue

            cur_t += 1
            if current_node.parent:
                current_node = current_node.parent
            else:
                process_one_step(cur_node = current_node)
                break

    deepest_leaf = find_deepest_leaf(node)
    # print(tree_to_expression(deepest_leaf))

    find_and_update(deepest_leaf)

    return steps

def find_deepest_leaf(node):
    if node.left.is_leaf and node.right.is_leaf:
        return node.left
    return find_deepest_leaf(node.right if node.left.is_leaf else node.left)

def generate_data(num_samples: int, T: int, k: int, t: int, mix: bool, model_size:int) -> List[dict]:
    data = []
    for _ in tqdm.tqdm(range(num_samples)):
        # random_T = random.sample(list(range(1, 4)) + list(range(4, T + 1, 4)),1)[0] if mix else T
        random_T = random.sample(list(range(12, T + 1)),1)[0] if mix else T
        tree = generate_tree(random_T, k)
        question = tree_to_expression(tree)
        answer = ff_evaluate(tree, k)
        cot_steps = generate_cot_steps(tree, k, opt_map[model_size][random_T])
        
        data.append({
            "depth": random_T,
            "ops_per_step":t-1,
            "question": question,
            "answer": str(answer),
            "cot": cot_steps
        })
    
    return data

def write_jsonl(data: List[dict], filename: str, write_type = 'w'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, write_type, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def parallel_generate_data(args):
    num_samples, T, k, t, test , model_size= args
    return generate_data(num_samples, T, k, t, test,model_size)


def gen_mixed_data(model_size):
    T = 80  # max depth of tree
    t = 12
    base_samples = 200                                                                   
    num_cpus = 10
    train_samples = num_cpus*base_samples*9
    val_samples = num_cpus*base_samples
    
    # ff_mod = 10  # 模运算的基数
    num_processes = 9  # 可根据实际机器的CPU核心数调整
    # ts = [1,2,4,8,16,32,64,128]

    for temp_t in ts[:t]:
        temp_t +=1
        args = [(train_samples // num_processes, T, ff_mod, temp_t, True,model_size) for _ in range(num_processes)]
        with Pool(num_processes) as pool:
            results = pool.map(parallel_generate_data, args)
        
        # Combine results and write to file
        train_data = [item for sublist in results for item in sublist]
        if temp_t == ts[0]+1:
            write_jsonl(train_data, f'data/arithmetic/{T}/mixed_t_{t}/train_{model_size}.jsonl')
        else:
            write_jsonl(train_data, f'data/arithmetic/{T}/mixed_t_{t}/train_{model_size}.jsonl', write_type = 'a')

        args = [(val_samples // num_processes, T, ff_mod, temp_t, True,model_size) for _ in range(num_processes)]
        with Pool(num_processes) as pool:
            results = pool.map(parallel_generate_data, args)
        
        # Combine results and write to file
        val_data = [item for sublist in results for item in sublist]
        if temp_t == ts[0]+1:
            write_jsonl(val_data, f'data/arithmetic/{T}/mixed_t_{t}/val_{model_size}.jsonl')
        else:
            write_jsonl(val_data, f'data/arithmetic/{T}/mixed_t_{t}/val_{model_size}.jsonl', write_type = 'a')

    prepare_and_save_data(f'data/arithmetic/{T}/mixed_t_{t}/', f'train_{model_size}', arithmeticTokenizer)
    prepare_and_save_data(f'data/arithmetic/{T}/mixed_t_{t}/', f'val_{model_size}', arithmeticTokenizer)



if __name__ == "__main__":
    for model_size in [5,6,7,8,9]:
        gen_mixed_data(model_size)



