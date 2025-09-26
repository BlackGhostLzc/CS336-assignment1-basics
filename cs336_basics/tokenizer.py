import os
from typing import BinaryIO, Iterator, Iterable
import regex as re
import multiprocessing
from collections import defaultdict


def display_file_info(file: BinaryIO):
    file.seek(0, os.SEEK_END)
    file_size = file.tell() 
    print(f"文件大小为 {file_size} 字节。")
    file.seek(0)


def display_pair_count(pair_counts):
    """
        对字节对按频率排序，并以可读的字符形式显示排名前10的结果。
    """
    sorted_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)
    top_10_pairs = sorted_pairs[:50]

    for rank, (pair, count) in enumerate(top_10_pairs, 1):
        char1 = pair[0].decode('utf-8', errors='replace')
        char2 = pair[1].decode('utf-8', errors='replace')
        char_representation = repr(char1 + char2)

        print(f"  Rank {rank}: Pair: {pair} -> {char_representation}, Count: {count}")
    



def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    f: BinaryIO = open(input_path, "rb")
    # display_file_info(f)
    num_processes = 4
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


    merge_epoch = (vocab_size - len(vocab))

    # 用来存储预分词后的的 chunk
    chunk_task_queue = multiprocessing.Queue()
    processes = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # 预分词用多进程共同完成
        process = multiprocessing.Process(
            target=worker, args=(chunk, vocab2idx, chunk_task_queue, special_tokens))
        
        processes.append(process)
        process.start() # 启动进程


    pair_counts: dict[tuple[int, int], int] = {} 

    processed_chunk_number = 0
    process_started_num = len(boundaries) - 1

    pretokens_ids = []
    # 需要一个map记录 pair -> index，也就是预分词后 token 的索引
    map_pair2index: dict[tuple[int, int], set] = defaultdict(set)

    while processed_chunk_number < process_started_num :
        # [[1,2..], [3,4..] ...]
        text_ids = chunk_task_queue.get()
        processed_chunk_number += 1

        # text = [], 对text[i]单独进行词元内的计数和merge，计数到counts中, text[i] 是预分词后的一个字节流
        for ids in text_ids:
            # 首先还需要对t进行 encode
            pretokens_ids.append(ids)
            index = len(pretokens_ids) - 1
            for pair in zip(ids, ids[1:]):
                pair_key = (pair[0], pair[1])
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
                map_pair2index[(pair[0], pair[1])].add(index)

    '''
       pretokens_ids : [[1,2..], [3,4..] ...] 
    '''


    for p in processes:
        p.join()
    
    # 这是所有的文本的 pretokens_bytes 列表
    '''
        [b'iron', b' cement', b' is', b' a', b' ready' ......................]
    '''

    # display_pair_count(bytes_pair_counts)

    # 计算好了所有的pair对的计数，然后就开始记录需要添加哪个词汇表，然后再做出merge
    for i in range(merge_epoch):
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





# CHANGE 1: 这是一个新的辅助函数，用于正确地按特殊符号切分文本
def split_by_special_tokens(text, special_tokens):
    """
    使用特殊符号切分文本，但保留特殊符号作为独立的块。
    """
    if text is None:
        return []
    
    if not special_tokens:
        return [text]
    
    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    # 使用 capturing group (...) 来在切分结果中保留分隔符
    chunks = re.split(f'({special_pattern})', text)
    # 过滤掉 re.split 可能产生的空字符串
    return [chunk for chunk in chunks if chunk]


def pre_tokenization(text, vocab2idx):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # 使用 regex.finditer
    str_tokens = [match.group() for match in re.finditer(PAT, text)]

    # 使用嵌套的列表推导式，代码更简洁
    result = [
        [vocab2idx[bytes([byte_val])] for byte_val in token.encode('utf-8')]
        for token in str_tokens
    ]
    return result


# CHANGE 2: 重写 worker 函数，采用正确的策略
def worker(text, vocab2idx, queue: multiprocessing.Queue, special_tokens: list[str]):
    """
    修正后的 worker 函数：不再移除特殊符号，而是用它们来切分文本并分别处理。
    """
    all_pretoken_ids = []
    
    # 1. 使用特殊符号进行切分，但保留它们
    text_chunks = split_by_special_tokens(text, special_tokens)
    
    # 2. 遍历所有切分块
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
            
    # 3. 将最终处理好的、结构正确的ID列表放入队列
    queue.put(all_pretoken_ids)



def find_top_pair(pair_counts, vocab):
    """
        从频率字典中找出计数最高的词对。
        pair_counts (dict): 格式为 {(token1, token2): count} 的字典。
        返回: tuple: 频率最高的词对, e.g., (104, 101)。如果字典为空则返回 None。
    """
    # 使用 max 函数和 key 参数可以高效地找到值最大的键，还有字典序比较
    return max(
        pair_counts, 
        key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]])
    )


def merge_and_update_counts(tokens, pair_to_merge, map_pair2index, pair_counts, new_token_id):
    id1, id2 = pair_to_merge

    # 根据 map_pair2index ，从 token_chunks 中找到相关的 index
    idset = map_pair2index[pair_to_merge]

    for index in idset:
        token = tokens[index]
        new_token = []
        # token: [1,2,3,4 ...... ]

        i = 0
        # for i in range(len(token) - 1):
        while i < len(token):
            # 检查当前元素和下一个元素是否匹配 id1 和 id2
            if ((i + 1) < len(token)) and token[i] == id1 and token[i+1] == id2:
                if i > 0:
                    pre_id = token[i-1]
                    downkey = (pre_id, id1)
                    pair_counts[downkey] = pair_counts.get(downkey, 0) - 1

                    upkey = (pre_id, new_token_id)
                    pair_counts[upkey] = pair_counts.get(upkey, 0) + 1
                    map_pair2index[upkey].add(index)

                if i + 2 < len(token):
                    post_id = token[i+2]
                    downkey = (id2, post_id)
                    pair_counts[downkey] = pair_counts.get(downkey, 0) - 1

                    upkey = (new_token_id, post_id)
                    pair_counts[upkey] = pair_counts.get(upkey, 0) + 1
                    map_pair2index[upkey].add(index)
                
                new_token.append(new_token_id)
                i += 2

            else:
                new_token.append(token[i])
                i += 1

        tokens[index] = new_token
    
    return tokens, pair_counts, map_pair2index


# 用于文件切分，并行处理？
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))





class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        '''
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 创建一个从 bytes 到 int 的反向映射
        self.vocab_bytes2int = {v: k for k, v in self.vocab.items()}

        # 这是一个合并的顺序排名
        self.bpe_ranks = {}
        for i, (p1, p2) in enumerate(merges):
            # 找到两个字节对应的 ID
            id1 = self.vocab_bytes2int[p1]
            id2 = self.vocab_bytes2int[p2]
            
            # 将 (ID对) 映射到它们的排名 i
            # 列表中的索引 i 越小，优先级越高
            self.bpe_ranks[(id1, id2)] = i


    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
        '''
            Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
            (in the same format that your BPE training code output) and (optionally) a list of special
            tokens. This method should accept the following additional parameters:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        '''
        pass

    
    def encode(self, text: str) -> list[int]:
        '''
            Encode an input text into a sequence of token IDs.
        '''
        '''
            a b c d <|endoftext|> e f g <|endoftext|> h i j <|endoftext|> k l m n <|endoftext|>
            使用一个正则表达式，根据你定义的所有特殊词元（如 <|endoftext|>),
            将输入的完整文本 text 切分成一个片段列表。
            "hello<|endoftext|>world" 会被切分成 ["hello", "<|endoftext|>", "world"]
        '''
        text_chunks = split_by_special_tokens(text, self.special_tokens)

        encode_ids = []
        # 遍历所有切分块
        for chunk in text_chunks:
            if self.special_tokens is not None and chunk in self.special_tokens:
                # 2a. 如果这个块是特殊符号，它本身就是一个完整的词元
                special_token_bytes = chunk.encode('utf--8')
                encode_ids.append(self.vocab_bytes2int[special_token_bytes])
            else:
                # 2b. 如果是普通文本
                ids = self.encodetext2ids(chunk, False)
                encode_ids.extend(ids)

        return encode_ids
    

    def encodetext2ids(self, chunk, is_special):
        pre_tokens = []
        # 如果是纯文本，不是特殊符号则需要预分词
        if is_special == False:
            pre_tokens = re.findall(self.PAT, chunk)
        else:
            pre_tokens = chunk
        chunk_ids = []

        for pre_token in pre_tokens:
            # 将单元字符串编码为字节序列
            token_bytes = pre_token.encode('utf-8')
            
            ids = [self.vocab_bytes2int.get(bytes([b])) for b in token_bytes]

            while len(ids) > 1:
                pairs = list(zip(ids, ids[1:]))

                best_pair_to_merge = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                if self.bpe_ranks.get(best_pair_to_merge) is None:
                    break
                
                id1, id2 = best_pair_to_merge

                for i in range(len(ids) - 1):
                    if ids[i] == id1 and ids[i+1] == id2:
                        new_bytes = self.vocab[id1] + self.vocab[id2]
                        new_id = self.vocab_bytes2int[new_bytes]
                        ids = ids[:i] + [new_id] + ids[i+2:]
                        break
            
            chunk_ids.extend(ids)
        
        return chunk_ids
        


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
            Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
            required for memory-efficient tokenization of large files that we cannot directly load into
            memory.
        '''
        pass


    def decode(self, ids: list[int]) -> str:
        '''
            Decode a sequence of token IDs into text.
        '''
        token_bytes = [self.vocab.get(id, b'') for id in ids]
        full_byte_sequence = b"".join(token_bytes)
        text = full_byte_sequence.decode('utf-8', errors='replace')

        '''
            text = ['Hello', ',', ' how', ' ', '<|endoftext|>', '<|endoftext|>', ' are', ' you', '?', '<|endoftext|>']
        '''

        return text
