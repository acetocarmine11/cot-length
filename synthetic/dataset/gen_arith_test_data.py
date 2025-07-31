import random
import json
from typing import List
import tqdm
import os
import sys
sys.path.append('..')
from ..tokenizor import ff_mod
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
        return f"({left}*{right})"

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
                # "expression_before": tree_to_expression(node),
                # "expression_after": None  # 将在更新树后填充
            }
            new_node = Node(value, parent = cur_node.parent)
            # step["expression_after"] = tree_to_expression(replace_node(node, cur_node, new_node))
            steps.append(step)

            #还需要修改parent的指针索引
            replace_node_in_parent(target = cur_node, new_node = new_node)
        
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

def generate_data(num_samples: int, T: int, k: int, t: int, test: bool) -> List[dict]:
    data = []
    for _ in tqdm.tqdm(range(num_samples)):
        random_T = T
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



if __name__ == "__main__":
    T = 80  # max depth of tree
    t = 12  # one step cot length
    test_samples_each = 100
    # data = generate_data(test_samples_each, 32, ff_mod, t, True)
    # write_jsonl(data, f'data/arithmetic/T_{32}.jsonl')
    
    for temp_T in range(t,T+1):
        data = generate_data(test_samples_each, temp_T, ff_mod, t, True)
        write_jsonl(data, f'data/arithmetic/test/new/{temp_T}.jsonl')