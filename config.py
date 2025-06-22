import torch
from pathlib import Path

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
IMG_DIR = Path("/usercode/flickr-8k/images")
TEXT_DIR = Path("/usercode/flickr-8k/text")

# Model hyperparameters
EMB_DIM = 768
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
ACTIVATION = "gelu"
MAX_LEN = 50

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 35
TRANSFORMER_LR = 1e-4
CNN_LR = 1e-5
CHECKPOINT_EPOCH = 5
EXP_DIR = "caption_generation_experimentation"