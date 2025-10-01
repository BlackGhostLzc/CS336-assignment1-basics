import torch
import argparse
import json
import sys
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- 添加结束 ---

from cs336_basics.model.transformer import TransformerLM
from cs336_basics.model.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.model.utils import save_checkpoint, load_checkpoint, cross_entropy

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' 
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / 'checkpoint' 
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'model.pth')

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='/home/lzc/assignment1-basics/cs336_basics/config.json', help='config file')
    parser.add_argument('--load', type=bool, default=False, help='load model from checkpoint')

    parser.add_argument('--iterations', type=int, default=40000, help='train model iterations')
    parser.add_argument('--save_interations', type=int, default=500, help='save model interval')
    parser.add_argument('--validate_interations', type=int, default=5000, help='save model interval')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context_length', type=int, default=256)

    parser.add_argument('--display_loss_iteration', type=int, default=1000)

    return parser.parse_args()


def get_memmap_dataset(path, dtype=np.int32):
    arr = np.memmap(path, dtype=dtype, mode="r")   # 单列token id序列
    return arr


def memmap_val_iterator(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    nb = (N-context_length-1)//batch_size
    for bi in range(nb // 20):
        base = bi*batch_size
        x = np.stack([memmap_arr[i:i+context_length] for i in range(base, base+batch_size)])
        y = np.stack([memmap_arr[i+1:i+context_length+1] for i in range(base, base+batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def get_batch(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    ix = np.random.randint(0, N-context_length-1, size=(batch_size,))
    x = np.stack([memmap_arr[i:i+context_length] for i in ix])
    y = np.stack([memmap_arr[i+1:i+context_length+1] for i in ix])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)



def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    
    # 定义模型并加载
    with open(args.config, 'r') as f:
        config = json.load(f)
    model_params = config["model"]
    model = TransformerLM(**model_params).to(device)
    
    # 优化器
    optimizer_params = config['optimizer']
    optimizer = AdamW(model.parameters())

    # 数据集
    train_dataset = get_memmap_dataset(os.path.join(DATA_DIR, "train.bin"))
    valid_dataset = get_memmap_dataset(os.path.join(DATA_DIR, "valid.bin"))

    
    # load 模型
    start_iter = 0
    if args.load == True:
        print("load model ... ")
        start_iter = load_checkpoint(CHECKPOINT_FILE, model, optimizer)
    

    display_loss_count = 0
    total_loss = 0
    MIN_LOSS = 1000.0

    for it in tqdm(range(start_iter, args.iterations), desc="Training"):
        model.train()

        # data, lables [batch_size, context_length]
        data, labels = get_batch(train_dataset, args.batch_size, args.context_length)
        data = data.to(device)
        labels = labels.to(device)

        # 清空梯度，设置学习率
        optimizer.zero_grad()
        lr = get_lr_cosine_schedule(
            it, optimizer_params['lr'], optimizer_params['min_lr'], optimizer_params['warmup_iters'], optimizer_params['cosine_iters']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向计算
        # logits [batch_size, context_length, vocab_size]
        logits = model(data)
        '''
            cross_entropy 参数：
                logits: Float[Tensor, " batch_size vocab_size"]
                targets: Int[Tensor, " batch_size"]
        '''
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.view(-1)
        loss = cross_entropy(logits, labels)
        total_loss += loss

        if display_loss_count == args.display_loss_iteration:
            avg_loss = total_loss / display_loss_count
            total_loss = 0
            display_loss_count = 0
            tqdm.write(f"iter {it:05d}: TRAIN loss = {avg_loss:.4f}")


        loss.backward()
        # 裁剪梯度
        gradient_clipping(model.parameters(), optimizer_params['clip_grad_norm'])

        optimizer.step()
        display_loss_count += 1


        if it % args.validate_interations == 0 and it > 0:
            model.eval()
            with torch.no_grad():
                val_losses = []

                N = len(valid_dataset)
                nb = (N - args.context_length - 1) // args.batch_size // 100

                val_iterator = memmap_val_iterator(valid_dataset, args.batch_size, args.context_length)
                progress_bar = tqdm(val_iterator, total=nb, desc="Validating", leave=False)
                
                for x_val, y_val in progress_bar:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits = model(x_val)
                    val_loss = cross_entropy(
                        val_logits.reshape(-1, val_logits.shape[-1]),
                        y_val.reshape(-1)
                    )
                    val_losses.append(val_loss.item())
    
                val_loss_mean = np.mean(val_losses)
                if val_loss_mean < MIN_LOSS:
                    save_checkpoint(model, optimizer, it, CHECKPOINT_FILE)
                    MIN_LOSS = val_loss_mean

                tqdm.write(f"iter {it:05d}: VALID loss = {val_loss_mean:.4f}")


if __name__ == '__main__':
    main()

