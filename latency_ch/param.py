import torch
from datetime import datetime


Hyper_Param = {
    'today': datetime.now().strftime('%Y-%m-%d'),
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tau': 0.001,
    'discount_factor': 0.95,
    'theta': 0.15,
    'dt': 0.01,
    'sigma': 0.2,
    'epsilon': 1.5,
    'epsilon_decay': 0.994,
    'epsilon_min': 0.0001,
    'lr_actor': 0.001,
    'lr_critic': 0.005,
    'batch_size': 2048,
    'train_start': 2200,
    'num_episode': 5000,
    'memory_size': 10**4,
    'print_every': 1,
    'num_neurons': [32,64,32,16],
    'critic_num_neurons': [32,32,16],
    # 'SensEnc_neurons' : [32],
    # 'SensDec_neurons' : [32],
    # 'SensEnc_latent_dim' : 10,
    # 'CUEnc_neurons' : [32],
    # 'CUDec_neurons' : [32],
    # 'CUEnc_latent_dim' : 4,
    'SensEnc_neurons' : [16],
    'SensDec_neurons' : [32],
    'SensEnc_latent_dim' : 4,
    'CUEnc_neurons' : [32],
    'CUDec_neurons' : [32],
    'CUEnc_latent_dim' : 4,
    'channel_type': 'fading',
    'SNR' : 5,
    'comm_latency': 0,  # ms
    '_iscomplex' : True,
    'Saved_using': False,
    'MODEL_PATH': "saved_model",
    'MODEL_NAME': "model_(227, 1001.0).h5"
}

