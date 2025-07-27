from transformers import AutoModelForCausalLM
import torch
from tokenizor import arithmeticTokenizer
import json
from contextlib import nullcontext
import torch
from tokenizor import arithmeticTokenizer, ts
import json
import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters for the model.")
    parser.add_argument('--test_T', type=int, default=60, help="The value of test_T (default: 32).")
    parser.add_argument('--test_t', type=int, default=4, help="The value of test_t (default: 2).")
    parser.add_argument('--t', type=int, default=8, help="The value of t (default: 5).")
    parser.add_argument('--model_size', type=int, default=9, help="The size of the model (default: 6).")
    # parser.add_argument('--each_head_dim', type=int, default=64, help="The dimension of each head (default: 64).")
    parser.add_argument('--T', type=int, default=60, help="The value of T (default: 32).")
    parser.add_argument('--device', type=int, default=0, help="The device to run the model on (default: cuda:0).")
    parser.add_argument('--caption', type=str, default='test', help="caption to result fine name")
    return parser.parse_args()


    

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
num_samples = 1 # number of samples to draw
max_new_tokens = 10000 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
args = get_args()
device = f'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
test_T = args.test_T
test_t = args.test_t
t = args.t
model_size = args.model_size
device_type = "cuda"
each_head_dim = 64
T = args.T
n_layer = model_size
n_head = model_size
n_embd = model_size * each_head_dim


# -----------------------------------------------------------------------------
# out_dir = f'out-arithmetic/model-size_{model_size}_{each_head_dim}/mixed_T_{T}/t_{t}' # ignored if init_from is not 'resume'
# out_dir = f'/workspace/stepwiseCoT/out-arithmetic/model-size_{model_size}/T_{T}/mixed_t_{t}'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


vocab_size = arithmeticTokenizer.vocab_len

# model = GPT2LMHeadModel(configuration)
model_path = "acetocarmine/M_6_T_80_t_12"
# out_dir = f'out-arithmetic/model-size_{model_size}/T_{T}/mixed_t_{t}'
model = AutoModelForCausalLM.from_pretrained(model_path)


model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)


encode = arithmeticTokenizer.encode
decode = arithmeticTokenizer.decode


if __name__ == "__main__":
    
    file_name = f"data/arithmetic/test/new/{test_T}.jsonl"
    data = []
    with open(file_name, 'r') as f:
        data = [json.loads(line) for line in f]
    batch_size = 10  # Define batch size for batching queries
    correct = 0
    total = 100
    prompt_nums = [0,0,0,0,0,0,0,0,0,0]

    # Process data in batches
    for i in tqdm.tqdm(range(0, total, batch_size)):
        batch = data[i:i + batch_size]
        questions = [sample['question'] + '<BEGIN>' + f"<{str(ts[test_t-1])}>"  for sample in batch]
        # questions = [sample['question'] + '<BEGIN>'  for sample in batch]
        encoded_questions = [torch.tensor(encode(q), dtype=torch.long, device=device)[None, ...] for q in questions]
        input_tensor = torch.cat(encoded_questions, dim=0)

        with torch.no_grad():
            with ctx:
                output_tokens = model.generate(inputs = input_tensor, max_new_tokens = 2500, eos_token_id=arithmeticTokenizer.end_token_id, do_sample=False, pad_token_id = arithmeticTokenizer.pad_token_id)#, top_k = 3)
                for idx, tokens in enumerate(output_tokens):
                    answer_tokens = tokens.tolist()
                    
                    index = answer_tokens.index(arithmeticTokenizer.end_token_id) if arithmeticTokenizer.end_token_id in answer_tokens else len(answer_tokens)
                    answer_str = decode(answer_tokens[:index])
                    correct += (batch[idx]['answer'] in answer_str[-3:])
                    prompt_idx = answer_str.index('>')
                    # print(answer_str[prompt_idx])
                    # print(answer_str)
                    prompt_nums[int(answer_str[prompt_idx+2])-1] +=1
                    if not (batch[idx]['answer'] in answer_str[-3:]):
                        print(answer_str[-3:])
                        print('---------------')
                        print(answer_str)
                        print(f"std_answer = {batch[idx]['answer']}")
                        print('---------------')


    output_file = f"eval_results/eval_results_{args.caption}.txt"
    print(correct/total)
    print(prompt_nums)
    with open(output_file, 'a') as out_file:
        out_file.write(f"model_size={model_size}, test_T={test_T}, t={test_t}, Accuracy: {correct/total:.4f}\n")