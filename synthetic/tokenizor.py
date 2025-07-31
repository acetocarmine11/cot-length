import json
import torch
import random
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import re
ts = [1,2,3,4,5,6,7,8,9,10,11,12]
class ArithmeticTokenizer:
    def __init__(self, k):
        self.vocab = [str(i) for i in range(k)] + ['+', '(', ')', '=', '\n',' ', '<END>', '<PAD>', '<BEGIN>'] + [f"<{str(t)}>" for t in ts]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id['<PAD>']
        self.end_token_id = self.token_to_id['<END>']
        self.begin_token_id = self.token_to_id['<BEGIN>']
        self.vocab_len = len(self.vocab)
        # 创建正则表达式来匹配所有token
        self.pattern = re.compile('|'.join(re.escape(token) for token in sorted(self.vocab, key=len, reverse=True)))

    def encode(self, text):
        tokens = self.pattern.findall(text)
        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

    def decode(self, ids):
        return ''.join([self.id_to_token[id] for id in ids if id in self.id_to_token])

class DPTokenizer:
    def __init__(self, k):
        self.vocab = [str(i) for i in range(k)] + ['\n',' ', '<END>', '<PAD>', '<BEGIN>'] + [f"<{str(t)}>" for t in ts]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id['<PAD>']
        self.end_token_id = self.token_to_id['<END>']
        self.begin_token_id = self.token_to_id['<BEGIN>']
        self.vocab_len = len(self.vocab)
        # 创建正则表达式来匹配所有token
        self.pattern = re.compile('|'.join(re.escape(token) for token in sorted(self.vocab, key=len, reverse=True)))

    def encode(self, text):
        tokens = self.pattern.findall(text)
        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

    def decode(self, ids):
        return ''.join([self.id_to_token[id] for id in ids if id in self.id_to_token])


def process_arithmetic_sample(sample, tokenizer):
    question = sample['question']
    cot = sample['cot']
    answer = sample['answer']
    ops_per_step = sample['ops_per_step']
    
    encoded_question = tokenizer.encode(question)
    # question_len = len(encoded_question) + 1 # should adjust the actual input question length
    question_len = len(encoded_question) #不含<k>
    processed = question + '<BEGIN>'+ f"<{str(ops_per_step)}>"  +'\n'
    for idx, step in enumerate(cot):
        processed += step['sub_expression'] + '=' + str(step['value']) + '\n'

    
    encoded_processed = tokenizer.encode(processed) + [tokenizer.end_token_id]
    
    return encoded_processed, question_len

def process_dp_sample(sample, tokenizer):
    question = sample['question']
    cot = sample['cot']
    ops_per_step = sample['ops_per_step']
    
    encoded_question = tokenizer.encode(question)
    question_len = len(encoded_question) #不含<k>
    processed = question + '<BEGIN>'+ f"<{str(ops_per_step)}>"  +'\n'
    for idx, step in enumerate(cot):
        processed += str(step['value']) + '\n'

    encoded_processed = tokenizer.encode(processed) + [tokenizer.end_token_id]
    
    return encoded_processed, question_len

def prepare_and_save_data(data_path, split, tokenizer, data_type = 'arithmetic'):
    # Read the data
    with open(data_path + f'{split}.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Process samples with progress bar
    processed_samples = []
    for sample in tqdm(data, desc=f"Processing {split} data", unit="sample"):
        if data_type == 'arithmetic':
            processed_samples.append(process_arithmetic_sample(sample, tokenizer))
        elif data_type == 'dp':
            processed_samples.append(process_dp_sample(sample, tokenizer))
    
    # Separate inputs and question lengths
    inputs = [sample[0] for sample in processed_samples]
    question_lengths = [sample[1] for sample in processed_samples]
    
    # Save inputs and labels
    torch.save(inputs, data_path + f'{split}_input.bin')
    torch.save(question_lengths, data_path + f'{split}_question_lengths.bin')
    
    print(f"Saved {split} data: {len(inputs)} samples")

def load_data(path, split):
    inputs = torch.load(path + f'{split}_input.bin', weights_only=True)
    question_lengths = torch.load(path + f'{split}_question_lengths.bin', weights_only=True)
    return inputs, question_lengths

def batch_generator(path, split, batch_size, block_size, device, tokenizer, ops_per_step = -1, t = 0):
    inputs, question_lengths = load_data(path, split)
    each_len = len(inputs)
    if t and  "val" in split and ops_per_step != -1:
        each_len = len(inputs)//t
        inputs = inputs[(ops_per_step)*each_len:(ops_per_step+1)*each_len]
        question_lengths = question_lengths[(ops_per_step)*each_len:(ops_per_step+1)*each_len]


    while True:
        # Randomly select indices for the batch
        batch_indices = random.sample(range(each_len), batch_size)
        
        # Get the selected inputs and question lengths
        batch_inputs = [inputs[i] for i in batch_indices]
        batch_question_lengths = [question_lengths[i] for i in batch_indices]
        
        # Convert to tensors and pad
        sequences = [torch.tensor(seq) for seq in batch_inputs]
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        # Truncate if necessary
        if padded_sequences.size(1) > block_size:
            padded_sequences = padded_sequences[:, :block_size]
        
        # Prepare input and target sequences
        x = padded_sequences[:, :-1].to(device)  # Input sequence
        y = padded_sequences[:, 1:].clone().to(device)  # Target sequence (shifted by 1)
        
        # Mask out the question part in y
        for i, length in enumerate(batch_question_lengths):
            y[i, :length] = tokenizer.pad_token_id
        
        yield x, y

ff_mod = 10  # Adjust as needed
arithmeticTokenizer = ArithmeticTokenizer(ff_mod)
dpTokenizer = DPTokenizer(ff_mod)


if __name__ == "__main__":

    T = 20
    t = 10
    path = f'synthetic/dataset/data/dp/{T}/mixed_t_{t}/'

    
    # # Example of getting a batch
    batch_size = 100
    block_size = 1024
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    train_batch = batch_generator(path, 'train', batch_size, block_size, device, dpTokenizer)
    x, y = next(train_batch)
    print("Input shape:", x.shape)
    print("Input shape:", dpTokenizer.decode(x[0].tolist()))

