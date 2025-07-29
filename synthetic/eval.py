from transformers import AutoModelForCausalLM
import torch
import json
from contextlib import nullcontext
import torch
from tokenizor import arithmeticTokenizer, ts, dpTokenizer
from model.looped_gpt2 import GPT, GPTConfig
import json
import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters for the model.")
    parser.add_argument('--test_T', type=int, default=60, help="The value of test_T (default: 32).")
    parser.add_argument('--test_t', type=int, default=4, help="The value of test_t (default: 2).")
    parser.add_argument('--t', type=int, default=8, help="The value of t (default: 5).")
    parser.add_argument('--model_size', type=int, default=9, help="The size of the model (default: 6).")
    parser.add_argument('--T', type=int, default=60, help="The value of T (default: 32).")
    parser.add_argument('--device', type=int, default=0, help="The device to run the model on (default: cuda:0).")
    parser.add_argument('--caption', type=str, default='test', help="caption to result fine name")
    parser.add_argument('--dp', action='store_true', help="Evaluate DP (dynamic programming) model")
    parser.add_argument('--model_type', type=str, default='gpt2', help="The model type to use (default: gpt2). Options: gpt2, looped_gpt2")
    parser.add_argument('--n_loop', type=int, default=1, help="The number of loops for looped_gpt2 model (default: 1).")
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
device = f'cuda' 
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

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


model_type = args.model_type
n_loop = args.n_loop

if args.dp:
    model_path = f'out-dp/model-size_{model_size}/T_{T}/mixed_t_{t}'
    tokenizor = dpTokenizer
else:
    model_path = "acetocarmine/M_6_T_80_t_12"
    tokenizor = arithmeticTokenizer

vocab_size = tokenizor.vocab_len

# Load model based on model_type
if model_type == 'looped_gpt2':
    # For looped_gpt2, we need to create the model from scratch and load state_dict
    configuration = GPTConfig(
        block_size=3100,  # Use the same block_size as in train.py
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=True,
        n_loop=n_loop
    )
    model = GPT(configuration)
    # Load the state_dict from the saved model
    state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
    model.load_state_dict(state_dict)
else:
    # Use standard AutoModelForCausalLM for gpt2 models
    model = AutoModelForCausalLM.from_pretrained(model_path)


model.eval()
model.to(device)
if compile:
    model = torch.compile(model) 



encode = tokenizor.encode
decode = tokenizor.decode



if __name__ == "__main__":
    
    if args.dp:
        file_name = f"dataset/data/dp/test/{test_T}.jsonl"
    else:
        file_name = f"dataset/data/arithmetic/test/{test_T}.jsonl"
    data = []
    with open(file_name, 'r') as f:
        data = [json.loads(line) for line in f]
    batch_size = 1  # Define batch size for batching queries
    correct = 0
    total = 10
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
                if model_type == 'looped_gpt2':
                    # For looped_gpt2 model, use the generate method from the GPT class
                    output_tokens = model.generate(input_tensor, max_new_tokens=2500, temperature=1.0, top_k=None)
                else:
                    # For standard gpt2 models, use the transformers generate method
                    output_tokens = model.generate(inputs = input_tensor, max_new_tokens = 2500, eos_token_id=tokenizor.end_token_id, do_sample=False, pad_token_id = tokenizor.pad_token_id)#, top_k = 3)
                for idx, tokens in enumerate(output_tokens):
                    answer_tokens = tokens.tolist()
                    
                    index = answer_tokens.index(tokenizor.end_token_id) if tokenizor.end_token_id in answer_tokens else len(answer_tokens)
                    answer_str = decode(answer_tokens[:index])
                    correct += (batch[idx]['answer'] in answer_str[-(len(batch[idx]['answer'])+1):])
                    prompt_idx = answer_str.index('>')
                    # print(answer_str[prompt_idx])
                    # print(answer_str)
                    prompt_nums[int(answer_str[prompt_idx+2])-1] +=1
                    print(answer_str[-(len(batch[idx]['answer'])+1):])
                    print('---------------')
                    print(answer_str)
                    print(f"std_answer = {batch[idx]['answer']}")
                    print('---------------')


    dataset_type = "dp" if args.dp else "arithmetic"
    output_file = f"eval_results/eval_results_{dataset_type}_{args.caption}_{model_type}_{n_loop}.txt"
    print(correct/total)
    print(prompt_nums)
    with open(output_file, 'a') as out_file:
        out_file.write(f"model_size={model_size}, model_type={model_type}, n_loop={n_loop}, test_T={test_T}, t={test_t}, Accuracy: {correct/total:.4f}\n")