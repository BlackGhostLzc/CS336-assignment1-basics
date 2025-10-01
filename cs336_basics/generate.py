import os
import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cs336_basics.model.transformer import TransformerLM
from cs336_basics.model.tokenizer import *
from cs336_basics.model.utils import load_checkpoint
from cs336_basics.model.optimizer import *
from cs336_basics.model.utils import softmax

from cs336_basics.train_bpe import load_tokenizer_from_vocab_merges_path

VOCAB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'vocab.json' 
MERGE_PATH = Path(__file__).resolve().parent.parent / 'data' / 'merges.txt' 
CONFIG_PATH = './config.json'
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / 'checkpoint' / 'model.pth'


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
):
    model.eval()
    token_ids = tokenizer.encode(prompt)            
    # tokens: [batch_size, seq_len]
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    '''
        如果没有KV Cache, 每次都需要把所有的 token_ids 输入 model 进行前向计算
        如果有KV Cache, 只需要在prefilling阶段把所有 token_ids 输入 model 进行前向计算, decoding阶段只需要输入上一轮新生成的 new_id
    '''
    endoftext_tokenbytes = "<|endoftext|>".encode('utf-8')

    print(f"输入提示: '{prompt}'")
    newtokens = []
    for _ in tqdm(range(max_new_tokens), desc="Generating"):
        logits = model(tokens)        #[batch_size, seq_len, vocab_size]

        next_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        next_token_logits = next_token_logits / temperature

        probs = softmax(next_token_logits, dim=-1)

        next_token_id = torch.argmax(probs, dim=-1)


        if tokenizer.vocab[next_token_id.item()] == endoftext_tokenbytes:
            break
        newtokens.append(next_token_id.item())
        next_token_id = next_token_id.unsqueeze(0)

        tokens = torch.cat([tokens, next_token_id], dim=1)

    response = tokenizer.decode(newtokens)
    print(response)





def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 首先根据 vocab 得到 tokenizer
    vocab, merges = load_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGE_PATH)
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    # 2. 加载 model
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model_params = config["model"]
    model = TransformerLM(**model_params).to(device)
    optimizer = AdamW(model.parameters())

    load_checkpoint(CHECKPOINT_PATH, model, optimizer)

    prompt = "introduce yourself, if you are a college student and want to try something new, what will you do, "

    generate(model, tokenizer, prompt, device, 200, 1.0)



if __name__ == '__main__':
    main()