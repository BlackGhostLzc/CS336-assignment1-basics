import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    '''
    # 最高的起始索引位置：[0,1,2....len-1], -1是因为还要验证集
    high = len(dataset) - context_length - 1
    start_index = np.random.randint(low=0, high=high + 1, size=(batch_size,))

    data_slices = [dataset[i : i + context_length] for i in start_index]
    data = np.stack(data_slices)
    label_slices = [dataset[i + 1: i + 1 + context_length] for i in start_index]
    label = np.stack(label_slices)

    return (torch.from_numpy(data).long().to(device), torch.from_numpy(label).long().to(device))