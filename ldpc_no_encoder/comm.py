import torch
import torch.nn as nn

from DDPG_module.MLP import MultiLayerPerceptron as MLP
from param import Hyper_Param
from robotic_env import RoboticEnv

DEVICE = Hyper_Param['DEVICE']
class Channel:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex

    def ideal_channel(self, x):
        return x

    def awgn(self, x, seed,snr):
        torch.manual_seed(seed)
        x = x.clone()
        std = (10 ** (-snr / 10.) / 2) ** 0.5 if self._iscomplex else (10 ** (
                    -snr / 10.)) ** 0.5  # for complex xs.
        # noise = torch.randn_like(x).to(DEVICE) * std
        # y = x + noise
        # print(noise)
        y = x + torch.randn_like(x).to(DEVICE) * std
        return y


    def fading(self, x,seed, snr):
        torch.manual_seed(seed)
        if self._iscomplex:
            x_shape = x.shape
            _dim = x_shape[1] // 2
            _std = (10 ** (-snr / 10.) / 2) ** 0.5
            _mul = torch.abs(torch.randn(x_shape[0], 2) / (2 ** 0.5)).to(DEVICE)# should divide 2**0.5 here.
            x_ = x.clone()
            x_[:, :_dim] *= _mul[:, 0].view(-1, 1)
            x_[:, _dim:] *= _mul[:, 1].view(-1, 1)
            x = x_
        else:
            _std = (10 ** (-snr / 10.)) ** 0.5
            x = x * torch.abs(torch.randn(x.shape[0], 1)).to(x)
        y = x + torch.randn_like(x) * _std
        # print(_std)
        return y


class Digitalize:
    def __init__(self, quant_max, q_bit, M, device='cuda'):
        self.q_bit = q_bit  # num_bit을 정수로 변환
        self.M = M
        self.min_data = 0
        self.max_data = quant_max
        self.device = device
        self.ldpc = LDPC(M,q_bit)

        num_side = torch.sqrt(torch.tensor(2 ** M)).int().item()
        qam_points = torch.arange(-(num_side - 1), num_side, 2).float()  # [-3, -1, 1, 3] for 16-QAM
        norm_factor = torch.sqrt(torch.mean(torch.abs(qam_points)**2 * 2))  # 평균 파워 정규화
        self.qam_points = (qam_points / norm_factor).to(self.device)  # (M,) 형태로 유지


    def Quantization(self, data):
        q_level = 2 ** self.q_bit
        data_clipped = torch.clamp(data, min=self.min_data, max=self.max_data)
        scaled_data = (data_clipped - self.min_data) / (self.max_data - self.min_data) * (q_level - 1)
        quantized_data = torch.round(scaled_data).int()
        
        bits = ((quantized_data.unsqueeze(-1).to(torch.int32) >> torch.arange(self.q_bit - 1, -1, -1, device=self.device)) & 1).float()

        return bits

    def Modulation(self, data):
        M = self.M

        half_M = M // 2

        # Split the bit sequence into two parts to map to I/Q separately
        I_bits = data[:,:, :half_M]  # (batch, num_sensors, M/2)
        Q_bits = data[:,:, half_M:]  # (batch, num_sensors, M/2)

        I_ints = torch.sum(I_bits * (2 ** torch.arange(half_M - 1, -1, -1, device=self.device)), dim=2).int().squeeze(1) # (batch,num_sensors)
        Q_ints = torch.sum(Q_bits * (2 ** torch.arange(half_M - 1, -1, -1, device=self.device)), dim=2).int().squeeze(1) # (batch,num_sensors)

        I = self.qam_points[I_ints]
        Q = self.qam_points[Q_ints]

        qam_symbols = torch.cat((I, Q), dim=1)  # (batch, 2*num_sensors)

        return qam_symbols

    def Demodulation(self, symbols):
        M = self.M

        # print(symbols)
        # print(symbols.size())
        # print('------------------------------')
        # I와 Q 성분 분리 (1, N) 형태로 사용
        half_len = symbols.size(1) // 2
        I = symbols[:, :half_len]  # (batch_size, N)
        Q = symbols[:, half_len:]  # (batch_size, N)

        # I/Q를 가장 가까운 QAM 포인트로 매핑
        I_index = torch.argmin(torch.abs(I.unsqueeze(-1) - self.qam_points.view(1,1,-1)), dim=-1)  # (batch_size, N)
        Q_index = torch.argmin(torch.abs(Q.unsqueeze(-1) - self.qam_points.view(1,1,-1)), dim=-1)  # (batch_size, N)

        # 인덱스를 비트 시퀀스로 변환
        data = I_index * M + Q_index  # (batch_size, N)
        # data를 비트 시퀀스로 변환
        bit_sequence = ((data.unsqueeze(-1).to(torch.int32) >> torch.arange(M - 1, -1, -1,
                                                                            device=self.device)) & 1).float()

        return bit_sequence

    def Dequantization(self, bit_sequence):
        q_level = 2 ** self.q_bit
        quantized_data = torch.sum(bit_sequence * (2 ** torch.arange(bit_sequence.size(-1) - 1, -1, -1, device=self.device)),
            dim=-1)  # (batch_size, 20)

        data = (quantized_data.float() / (q_level - 1)) * (self.max_data - self.min_data) + self.min_data
        return data

    def Txapply(self, data):
        quantized_data = self.Quantization(data.to(self.device))

        encoded_data = self.ldpc.encode(quantized_data)

        modulated_symbols = self.Modulation(encoded_data)

        return modulated_symbols

    def Rxapply(self, symbols):
        demodulated_data = self.Demodulation(symbols.to(self.device))
        # print(demodulated_data)
        decoded_data = self.ldpc.decode(demodulated_data)
        # print(decoded_data)
        recovered_data = self.Dequantization(decoded_data)
        return recovered_data




class LDPC:
    def __init__(self, n, k, device='cuda'):
        self.n = n  # 총 비트 수
        self.k = k  # 메시지 비트 수
        self.r = n - k  # 패리티 비트 수
        self.device = device

        # H-행렬 생성: 간단한 희소 패리티 검사 행렬
        self.H = torch.randint(0, 2, (self.r, self.n), dtype=torch.float32, device=self.device)
        self.G = self.generate_g_matrix()


    def generate_g_matrix(self):
        """
        생성 행렬 G 생성: G = [I_k | P^T]
        """
        # 패리티 하위행렬 P 생성
        P = self.H[:, :self.k]
        I_k = torch.eye(self.k, dtype=torch.float32, device=self.device)
        G = torch.cat((I_k, P.T), dim=1)
        return G

    def encode(self, message):

        codeword = torch.matmul(message, self.G) % 2
        return codeword

    def decode(self, received, max_iter=10):
        batch_size, num_blocks, _ = received.shape
        decoded = received.clone().to(self.device)  # 초기화: 받은 메시지를 그대로 복사

        for _ in range(max_iter):
            # 패리티 검사
            parity_check = torch.matmul(decoded, self.H.T) % 2  # (batch_size, 20, r)
            errors = parity_check.sum(dim=-1) > 0  # 패리티 오류가 있는 블록 확인 (batch_size, 20)

            # 오류가 없는 경우 반복 종료
            if not errors.any():
                break

            # 오류 위치 수정 (비트 플리핑)
            error_counts = torch.matmul(parity_check.float(), self.H)  # 각 비트가 오류에 관여한 횟수 계산 (batch_size, 20, n)
            flip_locations = error_counts > (self.r // 2)  # 오류 관여 횟수가 절반 이상인 비트를 플리핑
            decoded[flip_locations] = 1 - decoded[flip_locations]  # 비트 플리핑

        return decoded[:, :, :self.k]  # (batch_size, 20, k) 형태의 정보 비트 반환

    def check_parity(self, codeword):

        # H * codeword^T == 0 이면 모든 패리티 조건이 만족됨
        parity_check = torch.matmul(codeword, self.H.T) % 2
        return torch.all(parity_check == 0, dim=1)







