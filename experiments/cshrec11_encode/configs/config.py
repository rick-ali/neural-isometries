# Logging Info
WORK_DIR = "" # Leave empty
PROJECT_NAME = "NISO_CSHREC11_ENCODE"

SHIFT = 0.2 
CSHREC11_DIR = ''

# Transformation parameters
ALPHA_EQUIV = 5.0e-1
BETA_MULT = 1.0e-2 

# Latent parameters
KERNEL_OP_DIM = 64
LATENT_DIM  = 64

BATCH_SIZE      = 8
TRUE_BATCH_SIZE = 8 
MAX_BATCH_SIZE  = 64

INPUT_SIZE = (96, 192) 

DATASET_NAME = "cshrec11"
DATA_COMPRESSION_TYPE = 'None' #None
DATA_TRAIN_SIZE = 26 * 20 * 30
DATA_TEST_SIZE = 26 * 30 * 4 

# Training parameters
CONV_ENC_CHANNELS    = (32, 64, 128, 256)
CONV_ENC_BLOCK_DEPTH = (2, 2, 2, 2)

CONV_DEC_CHANNELS    = (256, 128, 64, 32)
CONV_DEC_BLOCK_DEPTH = (2, 2, 2, 2) 

KERNEL_SIZE = 3

## Hyperparameters

ADAM_B1 = 0.9 #0.9# 0.5
ADAM_B2 = 0.999 #0.999 # 0.9 

NUM_TRAIN_STEPS = 1000000
STOP_AFTER      = 50000
INIT_LR         = 1.0e-8 
LR              = 5.0e-4 
END_LR          = 5.0e-5
WARMUP_STEPS    = 2000 

EVAL_EVERY     = 5000
LOG_LOSS_EVERY = 100
VIZ_EVERY      = 2500
VIZ_SIZE       = INPUT_SIZE
NUM_EVAL_STEPS = 10 

CHECKPOINT_EVERY = 10000


TRIAL = 0  # Dummy for repeated runs.

