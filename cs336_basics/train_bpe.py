import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import heapq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cs336_basics.model.tokenizer import *

# 1. 在运行 train.py 之前，首先要运行这个文件把 TinyStoriesV2-GPT4 这个数据集的分词器训练好
# 2. 然后再把数据集文本用 vocab id 来表示, 序列化进磁盘 train.bin   valid.bin

data_dir = Path(__file__).resolve().parent.parent / 'data' 
train_text_path = os.path.join(data_dir, 'TinyStoriesV2-GPT4-train.txt')
valid_text_path = os.path.join(data_dir, 'TinyStoriesV2-GPT4-valid.txt')


worker_data_dir = Path(__file__).resolve().parent.parent / 'data' / 'TinyStoriesV2-GPT4-Worker' 


# 纯文本文件encode成int的文件
train_bin_path = os.path.join(data_dir, 'train.bin')
valid_bin_path = os.path.join(data_dir, 'valid.bin')

# 保存 vocab 和 merges 的文件
vocab_path = os.path.join(data_dir, 'vocab.json')
merges_path = os.path.join(data_dir, 'merges.txt')

# 搬运自 tests/common.py 文件
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def save_tokenizer_to_vocab_merges_path(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
):
    """
    Serializes a vocab and merges list to disk in the GPT-2 format.
    """
    # Create the mapping from a single byte to its GPT-2 unicode representation.
    gpt2_unicode_encoder = gpt2_bytes_to_unicode()

    # --- 1. Process and save the vocabulary ---
    vocab_to_save = {}
    for token_id, byte_sequence in vocab.items():
        # Convert the byte sequence into its corresponding GPT-2 unicode string
        gpt2_token = "".join([gpt2_unicode_encoder[b] for b in byte_sequence])
        vocab_to_save[gpt2_token] = token_id
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {vocab_path}")

    # --- 2. Process and save the merges ---
    with open(merges_path, "w", encoding="utf-8") as f:
        for token1_bytes, token2_bytes in merges:
            # Convert each part of the merge rule to its unicode string representation
            gpt2_token1 = "".join([gpt2_unicode_encoder[b] for b in token1_bytes])
            gpt2_token2 = "".join([gpt2_unicode_encoder[b] for b in token2_bytes])
            # Write the formatted line to the file
            f.write(f"{gpt2_token1} {gpt2_token2}\n")
    print(f"Merges saved to {merges_path}")




def load_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Loads a vocab and merges list from disk in the GPT-2 format.
    This is the inverse operation of save_tokenizer_to_vocab_merges_path.
    """
    # --- 1. Create the reverse mapping from unicode characters back to bytes ---
    gpt2_unicode_encoder = gpt2_bytes_to_unicode()
    gpt2_unicode_decoder = {v: k for k, v in gpt2_unicode_encoder.items()}

    # --- 2. Load and process the vocabulary ---
    print(f"Loading vocabulary from {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        loaded_vocab = json.load(f)
    
    vocab = {}
    for gpt2_token, token_id in loaded_vocab.items():
        # Convert the GPT-2 unicode string back into its original byte sequence
        byte_sequence = bytes([gpt2_unicode_decoder[char] for char in gpt2_token])
        vocab[token_id] = byte_sequence

    # --- 3. Load and process the merges ---
    print(f"Loading merges from {merges_path}")
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        # Skip the first line which is typically a version comment (e.g., "#version: 0.2")
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split the line into the two parts of the merge rule
            gpt2_token1, gpt2_token2 = line.split(" ")
            
            # Convert each part back to its byte representation
            token1_bytes = bytes([gpt2_unicode_decoder[char] for char in gpt2_token1])
            token2_bytes = bytes([gpt2_unicode_decoder[char] for char in gpt2_token2])
            
            merges.append((token1_bytes, token2_bytes))
            
    return vocab, merges



def train_bpe_tokenizer1(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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


    merge_epoch = (vocab_size - len(vocab))
    pair_counts: dict[tuple[int, int], int] = {} 
    pretokens_ids = []
    map_pair2index: dict[tuple[int, int], set] = defaultdict(set)
    numbers = 0

    for entry in worker_data_dir.iterdir():
        if numbers >= 1:
            break

        if entry.is_file():
            with open(os.path.join(worker_data_dir, entry), 'r', encoding='utf-8') as f:
                data = json.load(f)

                for ids in data:
                    pretokens_ids.append(ids)
                    index = len(pretokens_ids) - 1

                    for pair in zip(ids, ids[1:]):
                        pair_key = (pair[0], pair[1])
                        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
                        map_pair2index[(pair[0], pair[1])].add(index)
        numbers += 1


    # display_pair_count(bytes_pair_counts)
    pbar = tqdm(range(merge_epoch), desc="Training BPE Merges")
    # 计算好了所有的pair对的计数，然后就开始记录需要添加哪个到词汇表，然后再做出merge
    for i in pbar:
        # 1.找出最大的 pair 对
        top_pair = find_top_pair(pair_counts, vocab)
        id1, id2 = top_pair
        # print("合并的id是 ", id1, id2)
        bytes1 = vocab[id1]
        bytes2 = vocab[id2]
        new_bytes = bytes1 + bytes2
        merge.append((bytes1, bytes2))

        # 加入词汇表 bytes -> int
        new_token_id = len(vocab)
        vocab[new_token_id] = new_bytes
        vocab2idx[new_bytes] = new_token_id

        # # 加入合并表 bytes bytes, 还需要把这个 top_pair 从 bytes_pair_counts 中删除
        del pair_counts[top_pair]

        # 2.更新 bytes_pair_counts 字典
        pretokens_ids, pair_counts, map_pair2index = \
            merge_and_update_counts(pretokens_ids, top_pair, map_pair2index, pair_counts, new_token_id)
    return (vocab, merge)


def _encode_iterable(tokenizer, iterable):
    """
    We place tokenizer.encode_iterable into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    yield from tokenizer.encode_iterable(iterable)



def encode_text2ids(tokenizer, text_path: str | os.PathLike, bin_path: str | os.PathLike):
    with open(text_path) as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)
    arr = np.array(ids, dtype=np.int32)

    arr.tofile(bin_path)



def train_bpe():
    # vocab, merges = train_bpe_tokenizer(input_path=valid_text_path, \
    #                                     vocab_size=10000, special_tokens=["<|endoftext|>"])
    
    vocab, merges = train_bpe_tokenizer1(input_path=train_text_path, \
                                        vocab_size=10000, special_tokens=["<|endoftext|>"])

    
    # 还需要把 vocab, merges 写入磁盘文件
    save_tokenizer_to_vocab_merges_path(vocab, merges, vocab_path, merges_path)



def process_dataset():
    vocab, merges = load_tokenizer_from_vocab_merges_path(vocab_path, merges_path)
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    # 把文本文件转成 vocab id 并序列化进磁盘
    encode_text2ids(tokenizer, train_text_path, train_bin_path)
    # encode_text2ids(tokenizer, valid_text_path, valid_bin_path)




if __name__ == '__main__':
    #train_bpe()
    process_dataset()



