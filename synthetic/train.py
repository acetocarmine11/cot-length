"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import os
import time
import math
from contextlib import nullcontext
import torch._dynamo
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
from tokenizor import batch_generator, arithmeticTokenizer
from transformers import GPT2LMHeadModel, GPT2Config
from torch.optim import AdamW
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Parse hyperparameters for the model.")
    parser.add_argument('--t', type=int, default=8, help="The value of t (default: 5).")
    parser.add_argument('--model_size', type=int, default=6, help="The size of the model (default: 6).")
    parser.add_argument('--device', type=str, default='cuda:0', help="The dimension of each head (default: 64).")
    parser.add_argument('--T', type=int, default=128, help="The value of T (default: 32).")
    parser.add_argument('--iter', type=int, default=20000, help="The value of T (default: 32).")
    return parser.parse_args()
# -----------------------------------------------------------------------------
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 1
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'arithmetic'
wandb_run_name = 'mini-gpt'

dataset = 'arithmetic'
gradient_accumulation_steps = 1

args = get_args()
batch_size = 256
T = args.T
t = args.t
model_size = args.model_size
device = args.device # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

block_size = 3100 

# baby GPT model :)
dropout = 0.0

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = args.iter
lr_decay_iters = args.iter # make equal to max_iters usually
decay_lr = True # whether to decay the learning rate
min_lr = 1e-5 # learning_rate / 10 usually
weight_decay = 0.05
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

beta1 = 0.9
beta2 = 0.98 # make a bit bigger because number of tokens per iter is small

warmup_iters = 50 # not super necessary potentially

position_encoding='absolute' # add positional encoding $$rotary or absolute$$

each_head_dim = 64

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
# culculate global values
n_layer = model_size
n_head = model_size
n_embd = model_size * each_head_dim
out_dir = f'synthetic/out-arithmetic/model-size_{model_size}/T_{T}/mixed_t_{t}'
# out_dir = f'out-arithmetic/model-size_{model_size}_{each_head_dim}/mixed_T_{T}/t_{t}'
result_path = f"synthetic/logs/train_results_{t}_{model_size}_{max_iters}.txt"
os.makedirs(os.path.dirname(result_path), exist_ok=True)
with open(result_path, 'a') as out_file:
    out_file.write(out_dir)
print(f"n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, out_dir={out_dir}")
# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# batch generater
data_dir = f'synthetic/dataset/data/arithmetic/{T}/mixed_t_{t}/'
train_batch = batch_generator(data_dir, 'train', batch_size, block_size, device, arithmeticTokenizer)
# val_batch = batch_generator(data_dir, 'val', batch_size, block_size, device, arithmeticTokenizer)
val_batchs = [batch_generator(data_dir, 'val', batch_size, block_size, device, arithmeticTokenizer, ops_per_step = ops_per_step, t = t) for ops_per_step in range(t)]

def get_batch(split, ops_per_step = -1):
    if split == 'train':
        x, y = next(train_batch)
    elif ops_per_step != -1:
        x, y = next(val_batchs[ops_per_step])
    return x, y

# train ini cfg
iter_num = 0
best_val_loss = float('inf')

# attempt to derive vocab_size from the dataset
meta_vocab_size = arithmeticTokenizer.vocab_len


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, vocab_size=None, dropout=dropout, position_encoding=position_encoding) # start with model_args from command line


#################################################################### load model ########################################################
configuration = GPT2Config(activation_function = 'gelu',eos_token_id = 16, bos_token_id = 18, n_positions = 3100, n_layer=n_layer, n_head=n_head, n_embd=n_embd,n_inner = 4*n_embd, block_size=block_size, vocab_size=meta_vocab_size, dropout=dropout, position_encoding=position_encoding)
print(configuration)
# Initializing a model from the configuration
model = GPT2LMHeadModel(configuration)
model.to(device)
#################################################################### load model ########################################################

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda',enabled=(dtype == 'float16'))

# optimizer
# optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    sub_task_loss = {}
    in_sub_task_loss = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            losses = torch.zeros(eval_iters)
            sub_losses = torch.zeros(eval_iters)
            in_sub_losses = torch.zeros(eval_iters)
        else:
            losses = torch.zeros(t, eval_iters)
            sub_losses = torch.zeros(t, eval_iters)
            in_sub_losses = torch.zeros(t, eval_iters)
        for k in range(eval_iters):
            if split == 'val':
                for ops_per_step in range(t):
                    X, Y = get_batch(split, ops_per_step)
                    with ctx:
                        logits = model(input_ids = X)[0]
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=arithmeticTokenizer.pad_token_id)
                    losses[ops_per_step][k] = loss.item()
                    out['val'] = losses.mean(dim=1)
                    
                    # 计算子任务损失
                    equal_indices = (X == arithmeticTokenizer.token_to_id['=']).nonzero()[:][::1].permute(1,0).tolist()
                    even_indices = equal_indices
                    # print(arithmeticTokenizer.decode(X[even_indices].tolist()), arithmeticTokenizer.decode(Y[even_indices].tolist()))
                    sub_logits = logits[even_indices]
                    sub_targets = Y[even_indices]
                    sub_loss = F.cross_entropy(sub_logits, sub_targets)
                    sub_losses[ops_per_step][k] = sub_loss.item()
                    sub_task_loss['val'] = sub_losses.mean(dim=1)

                    #计算其他损失
                    inequal_indices = (X != arithmeticTokenizer.token_to_id['=']).nonzero()[:][::1].permute(1,0).tolist()
                    even_indices = inequal_indices
                    # print(arithmeticTokenizer.decode(X[even_indices].tolist()), arithmeticTokenizer.decode(Y[even_indices].tolist()))

                    
                    in_sub_logits = logits[even_indices]
                    in_sub_targets = Y[even_indices]
                    in_sub_loss = F.cross_entropy(in_sub_logits, in_sub_targets, ignore_index = arithmeticTokenizer.pad_token_id)
                    in_sub_losses[ops_per_step][k] = in_sub_loss.item()
                    in_sub_task_loss['val'] = in_sub_losses.mean(dim=1)
            else:
                X, Y = get_batch(split)
                with ctx:
                    logits = model(input_ids = X)[0]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=arithmeticTokenizer.pad_token_id)
                losses[k] = loss.item()
                out['train'] = losses.mean()
                
                # 计算子任务损失
                equal_indices = (X == arithmeticTokenizer.token_to_id['=']).nonzero()[:][::2].permute(1,0).tolist()
                even_indices = equal_indices

                sub_logits = logits[even_indices]
                sub_targets = Y[even_indices]
                sub_loss = F.cross_entropy(sub_logits, sub_targets)
                sub_losses[k] = sub_loss.item()
                sub_task_loss['train'] = sub_losses.mean()

                #计算其他损失
                inequal_indices = (X != arithmeticTokenizer.token_to_id['=']).nonzero()[:][::2].permute(1,0).tolist()
                even_indices = inequal_indices

                in_sub_logits = logits[even_indices]
                in_sub_targets = Y[even_indices]
                in_sub_loss = F.cross_entropy(in_sub_logits, in_sub_targets, ignore_index = arithmeticTokenizer.pad_token_id)
                in_sub_losses[k] = in_sub_loss.item()
                in_sub_task_loss['train'] = in_sub_losses.mean()
    
    model.train()
    
    print("Main task loss:", out)
    print("Sub-task loss:", sub_task_loss)
    print("In-Sub-task loss:", in_sub_task_loss)
    return out, sub_task_loss, in_sub_task_loss


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    # wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
save_flag = [True, True, True, True]
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:

        losses, sub_task_losses, in_sub_task_losses = estimate_loss()
        with open(result_path, 'a') as out_file:
            out_file.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']}, sub train loss {sub_task_losses['train']:.4f}, sub val loss {sub_task_losses['val']}, in sub train loss {in_sub_task_losses['train']:.4f}, in sub val loss {in_sub_task_losses['val']}, lr: {lr}\n")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'].mean() < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val'].mean()
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                model.save_pretrained(out_dir)
                save_path = "pytorch_model.bin"
                torch.save(model.state_dict(), out_dir + '/' + save_path)
                print(f"Model saved to {save_path}")

    # if iter_num == 0:
    #     break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            # logits, loss = model(X, Y)
            logits = model(input_ids = X)[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=arithmeticTokenizer.pad_token_id)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5: # let the training loop settle a bit
            # mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            # running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

    
if ddp:
    destroy_process_group()
