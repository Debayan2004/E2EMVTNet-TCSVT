
import os
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = os.environ.get(
    "E2EMVTNET_DATASET_ROOT",
    "/kaggle/input/datasets/debayan20000/icdar-2015-multiview/ICDAR2015_MultiView"
)

NUM_VIEWS = 4

DET_INPUT_HEIGHT = 256
DET_INPUT_WIDTH = 1024

REC_INPUT_HEIGHT = 64
REC_INPUT_WIDTH = 256

MAX_TEXT_LENGTH = 32

BATCH_SIZE = 2
ACCUMULATION_STEPS = 8
NUM_WORKERS = 1

EPOCHS = 1000

LEARNING_RATE = 3e-5
LOSS_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
USE_AMP = torch.cuda.is_available()

BASE_CHANNELS = 48
ENCODER_CHANNELS = [48, 96, 192, 384]
FEATURE_DIM = 192
MEMORY_DIM = 192

CHARSET = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)

PAD_IDX = 0
EOS_IDX = 1

CHAR2IDX = {
    character: index + 2
    for index, character in enumerate(CHARSET)
}

IDX2CHAR = {
    index + 2: character
    for index, character in enumerate(CHARSET)
}

VOCAB_SIZE = len(CHARSET) + 2

PAD_TOKEN = 0
