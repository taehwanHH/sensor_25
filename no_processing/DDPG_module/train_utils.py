import numpy as np
import torch
import argparse
from param import Hyper_Param

def to_tensor(np_array: np.array, size=None) -> torch.tensor:
    torch_tensor = torch.from_numpy(np_array).float()
    if size is not None:
        torch_tensor = torch_tensor.view(size)
    return torch_tensor


def to_numpy(torch_tensor: torch.tensor) -> np.array:
    return torch_tensor.cpu().detach().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Parse SNR and channel_type parameters")

    # argparse로 받을 파라미터들 정의 (snr과 channel_type만)
    parser.add_argument('--snr', type=int, default=Hyper_Param['SNR'], help='Signal-to-noise ratio')
    parser.add_argument('--channel_type', type=str, default=Hyper_Param['channel_type'], help='Type of communication channel')
    parser.add_argument('--latency', type=int, default=Hyper_Param['comm_latency'], help='Communication latency')
    parser.add_argument('--_iscomplex', type=bool, default=Hyper_Param['_iscomplex'], help='Channel complex')


    args = parser.parse_args()
    return args


class EMAMeter:

    def __init__(self,
                 alpha: float = 0.5):
        self.s = None
        self.alpha = alpha

    def update(self, y):
        if self.s is None:
            self.s = y
        else:
            self.s = self.alpha * y + (1 - self.alpha) * self.s
