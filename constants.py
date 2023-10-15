############################## Preprocessing #################################### 
root='/content/drive/MyDrive/TTS_2023_V3/'
CORPUS_PATH = root+"content/arabic-speech-corpus"
RAW_DATA_PATH = root+"data/raw_data"
DATA_PATH = root+"data/preprocessed_data"
 
VAL_SIZE = 2
# Audio
SAMPLING_RATE = 22050
MAX_WAVE_VALUE = 32768.0
# STFT
STFT_FILTER_LENGTH =  1024
STFT_HOP_LENGTH = 256
STFT_WIN_LENGTH = 1024
# Mel
N_MEL_CHANNELS = 80
MEL_FMIN = 0
MEL_FMAX = 8000  

########################### Train ############################
# Path:
CKPT_PATH = root+"output/ckpt"
LOG_PATH = root+"output/log"
RESULT_PATH = root+"output/result"
# Optimizer:
BATCH_SIZE = 1
BETAS = [0.9, 0.98]
EPS = 0.000000001
WEIGHT_DECAY = 0.0
GRAD_CLIP_THRESH= 1.0
GRAD_ACC_STEP= 1
WARM_UP_STEP= 4000
ANNEAL_STEP= [300000, 400000, 500000]
ANNEAL_RATE= 0.3
# Step:
TOTAL_STEP = 1045000
LOG_STEP = 1000
SYNTH_STEP =  1000
VAL_STEP= 10000
SAVE_STEP= 5000

VOCODER_CONFIG_PATH=root+'hifigan/config.json'
VOCODER_PRETRAINED_MODEL_PATH=root+'hifigan/generator_universal.pth.tar'


############################### MODEL ##############################
# transformer:
ENCODER_LAYER= 4
ENCODER_HEAD= 2
ENCODER_HIDDEN= 256
DECODER_LAYER= 6
DECODER_HEAD= 2
DECODER_HIDDEN= 256
CONV_FILTER_SIZE= 1024
CONV_KERNEL_SIZE= [9, 1]
ENCODER_DROPOUT= 0.2
DECODER_DROPOUT= 0.2

# variance_predictor:
FILTER_SIZE= 256
KERNEL_SIZE= 3
DROPOUT= 0.5

# variance_embedding:
N_BINS= 256

MAX_SEQ_LEN = 1000

# vocoder:
MODEL = "HiFi-GAN" 
SPEAKER = "universal" 

RESTORE_STEP=700000