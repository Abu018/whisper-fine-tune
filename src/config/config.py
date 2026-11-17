# Model
from pickle import TRUE


MODEL_ID = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"
MODEL_FILE = "indicconformer_stt_hi_hybrid_rnnt_large.nemo"
LANGUAGE_ID = "hi"
CONVERTED_MODEL_PATH = "models/indicconformer_hi.nemo"
LANGUAGE = "hindi"
TASK = "transcribe"
FREEZE_ENCODER = True


NUM_PROC=4
TOKEN=True
# Data
DATASET_ID = "ai4bharat/Kathbath"
DATASET_CONFIG = "hindi"
TRAIN_SPLIT = "train"
TRAIN_SAMPLES = 1800   # about 2% of the 91,752 training clips
EVAL_SPLIT = "valid"
EVAL_SAMPLES = 200
CACHE_CONFIG="/data/cache"
SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_LABEL_TOKENS = 448

# Training
EPOCHS = 10
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
LR_SCHEDULER = "linear"
MAX_GRAD_NORM = 1.0
FP16 = True
GRADIENT_CHECKPOINTING = True
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING_PATIENCE = 3

# Run
SEED = 42
OUTPUT_DIR = "outputs/t4_small"