import os
import torch

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = f"/home/mohit/Documents/project/interpolation/HiCInterpolate"
DATA_DIR = f"/home/mohit/Documents/project/interpolation/data/triplets/normalized"

RECORD_FILE = f"{DATA_DIR}/128/train.txt"
IMAGE_DIR = f"{DATA_DIR}/128"
INTERPOLATOR_IMAGES_MAP = {
    'frame_0': 'img1.npy',
    'frame_1': 'img2.npy',
    'frame_2': 'img3.npy'
}

TRAIN_VAL_RATIO = [0.85, 0.15]
BATCH_SIZE = 32

NUM_EPOCHS = 1000
SAVE_EVERY = 10
LEARNING_RATE = 0.0001
DECAY_STEPS = 50
DECAY_RATE = 0.95
LEARNING_RATE_STAIRCASE = True

INIT_IN_CHANNELS = 1
INIT_OUT_CHANNELS = 4

PYRAMID_LEVEL = 5  # Input pyramid level
EXT_FEATURE_LEVEL = 4  # Features stack goes to optical flow

UNIQUE_LEVELS = 3
FLOW_NUM_OF_CONVS = [3, 3, 3, 3]
FLOW_OUT_CHANNELS = [16, 32, 64, 128]

FUSION_PYRAMID_LEVEL = 5


VGG_MODEL_FILE = f"{DATA_DIR}/models/imagenet-vgg-verydeep-19.mat"
LOSS_WEIGHT_PARAMETERS = [
    {'name': 'l1', 'boundaries': [0], 'values': [1.0, 1.0]},
    {'name': 'vgg', 'boundaries': [500], 'values': [1.0, 0.25]},
    {'name': 'style', 'boundaries': [500], 'values': [0.0, 40.0]}]

NUM_VISUALIZATION_SAMPLES = 2
IMG_VAL_PLOT_PATH = os.path.sep.join(
    [ROOT_DIR, "output"])

MODEL_NAME = 'hicinterpolate_128'
MODEL_FILE = os.path.join(ROOT_DIR, "models", f"{MODEL_NAME}.pth")
TRAIN_VAL_PLOT_FILE = os.path.sep.join(
    [ROOT_DIR, "output", f"{MODEL_NAME}_train_val_plot.png"])
TEST_PLOT_FILE = os.path.sep.join(
    [ROOT_DIR, "output", f"{MODEL_NAME}_test_plot.png"])

EVAL_METRICS_FILE = os.path.join(
    ROOT_DIR, "output", f"{MODEL_NAME}_val_metrics.csv")
PSNR_EVAL_PLOT_FILE = os.path.join(
    ROOT_DIR, "output", f"{MODEL_NAME}_psnr_plot.png")
SSIM_EVAL_PLOT_FILE = os.path.join(
    ROOT_DIR, "output", f"{MODEL_NAME}_ssim_plot.png")
LR_FILE = os.path.join(ROOT_DIR, "output", f"{MODEL_NAME}_lr_plot.png")
MODEL_STATE_DIR = f"{ROOT_DIR}/models"
