import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cs336_basics.model.tokenizer import *


# 在 tokenizer.py 的实现中，把共享队列换成磁盘文件
# 由 master 逐个遍历磁盘文件统计 pair count 进行 train bpe


data_dir = Path(__file__).resolve().parent.parent / 'data'
input_path = os.path.join(data_dir, 'TinyStoriesV2-GPT4-train.txt')
special_tokens=["<|endoftext|>"]

out_dir = os.path.join(data_dir, 'TinyStoriesV2-GPT4-Worker')

def worker1(text, vocab2idx, special_tokens: list[str], process_id):
    all_pretoken_ids = []

    text_chunks = split_by_special_tokens(text, special_tokens)
    for chunk in text_chunks:
        if chunk in special_tokens:
            # 2a. 如果这个块是特殊符号，它本身就是一个完整的词元
            special_token_bytes = chunk.encode('utf--8')
            if special_token_bytes in vocab2idx:
                special_token_id = vocab2idx[special_token_bytes]
                all_pretoken_ids.append([special_token_id])
        else:
            # 2b. 如果是普通文本，对其进行预分词
            pretoken_ids_list = pre_tokenization(chunk, vocab2idx)
            all_pretoken_ids.extend(pretoken_ids_list)

    filename = f"chunk_{process_id}.json"
    output_path = os.path.join(out_dir, filename)
    print(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pretoken_ids, f)
    


def main():
    f: BinaryIO = open(input_path, "rb")
    num_processes = 8
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 初始化 vocab 
    vocab: dict[int, bytes] = {}
    merge: list[tuple[bytes, bytes]] = []

    vocab2idx: dict[bytes, int] = {}

    # 首先把特殊符号加入词汇表
    for special_token in special_tokens:
        token_bytes = special_token.encode('utf-8')
        token_id = len(vocab)
        vocab[token_id] = token_bytes
        vocab2idx[token_bytes] = token_id


    # 添加 256 个初始的字节
    for i in range(256):
        token_bytes = bytes([i])
        token_id = len(vocab)
        vocab[token_id] = token_bytes
        vocab2idx[token_bytes] = token_id


    process_id = 0
    processes = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # 预分词用多进程共同完成
        process = multiprocessing.Process(
            target=worker1, args=(chunk, vocab2idx, special_tokens, process_id))
        
        processes.append(process)
        process.start() # 启动进程
        process_id += 1

    for p in processes:
        p.join()





if __name__ == '__main__':
    main()