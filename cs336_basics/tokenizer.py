import os
from typing import BinaryIO, Iterator
import regex as re
import multiprocessing



def display_file_info(file: BinaryIO):
    file.seek(0, os.SEEK_END)
    file_size = file.tell() 
    print(f"文件大小为 {file_size} 字节。")
    file.seek(0)



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

    for special_token in special_tokens:
        token_bytes = special_token.encode('utf-8')
        token_id = len(vocab)
        vocab[token_id] = token_bytes


    # 添加 256 个初始的字节
    for i in range(256):
        token_byte = bytes([i])
        token_id = len(vocab)
        vocab[token_id] = token_byte

    merge_epoch = (vocab_size - len(vocab))

    chunk_list = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk_list.append(chunk)


    for chunk in chunk_list:
        pass

        
    return vocab, None




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
        pass


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
        pass


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
        pass