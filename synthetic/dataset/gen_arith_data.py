import random
import json
from typing import List, Union
import tqdm
import os
import sys
from ..tokenizor import prepare_and_save_data, arithmeticTokenizer, ff_mod, ts
from multiprocessing import Pool
import multiprocessing

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

def generate_data(num_samples: int, T: int, k: int, t: int, mix: bool) -> List[dict]:
    data = []
    for _ in tqdm.tqdm(range(num_samples)):
        # random_T = random.sample(list(range(1, 4)) + list(range(4, T + 1, 4)),1)[0] if mix else T
        random_T = random.sample(list(range(12, T + 1)),1)[0] if mix else T
        tree = generate_tree(random_T, k)
        question = tree_to_expression(tree)
        answer = ff_evaluate(tree, k)
        cot_steps = generate_cot_steps(tree, k, t)
        
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
    num_samples, T, k, t, test = args
    return generate_data(num_samples, T, k, t, test)


def gen_mixed_data():
    T = 80  # max depth of tree
    t = 12
    base_samples = 200                                                                   
    num_cpus = 10
    train_samples = num_cpus*base_samples*9
    val_samples = num_cpus*base_samples 
    num_processes = 9  

    for temp_t in ts[:t]:
        temp_t +=1
        args = [(train_samples // num_processes, T, ff_mod, temp_t, True) for _ in range(num_processes)]
        with Pool(num_processes) as pool:
            results = pool.map(parallel_generate_data, args)
        
        # Combine results and write to file
        train_data = [item for sublist in results for item in sublist]
        if temp_t == ts[0]+1:
            write_jsonl(train_data, f'synthetic/dataset/data/arithmetic/{T}/mixed_t_{t}/train.jsonl')
        else:
            write_jsonl(train_data, f'synthetic/dataset/data/arithmetic/{T}/mixed_t_{t}/train.jsonl', write_type = 'a')

        args = [(val_samples // num_processes, T, ff_mod, temp_t, True) for _ in range(num_processes)]
        with Pool(num_processes) as pool:
            results = pool.map(parallel_generate_data, args)
        
        # Combine results and write to file
        val_data = [item for sublist in results for item in sublist]
        if temp_t == ts[0]+1:
            write_jsonl(val_data, f'synthetic/dataset/data/arithmetic/{T}/mixed_t_{t}/val.jsonl')
        else:
            write_jsonl(val_data, f'synthetic/dataset/data/arithmetic/{T}/mixed_t_{t}/val.jsonl', write_type = 'a')

    prepare_and_save_data(f'synthetic/dataset/data/arithmetic/{T}/mixed_t_{t}/', 'train', arithmeticTokenizer)
    prepare_and_save_data(f'synthetic/dataset/data/arithmetic/{T}/mixed_t_{t}/', 'val', arithmeticTokenizer)

    

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) 
    gen_mixed_data()



