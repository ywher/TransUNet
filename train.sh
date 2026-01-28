

# Dataset settings
DATASET=camus              # choose between camus / synapse
ROOT_PATH=./data           # dataset root; for synapse use ../data/Synapse/train_npz
LIST_DIR=./lists/camus     # list files root
CAMUS_SPLIT=all            # split folder under lists/camus (e.g., all, 1_2, 1_4, 1_8, 1_16, 1_20, 1_100)

# Model configuration
VIT_NAME=R50-ViT-B_16      # backbone variant
VIT_PATCHES_SIZE=16        # patch size for ViT
N_SKIP=3                   # number of skip connections
IMG_SIZE=512               # input size

# Training hyperparameters
MAX_ITERATIONS=30000
MAX_EPOCHS=180
BATCH_SIZE=8
BASE_LR=0.01
N_GPU=1

# Determinism / seeding
DETERMINISTIC=1            # 1 enables deterministic mode
SEED=1234

# Validation / early stopping
EVAL_INTERVAL=5            # evaluate every N epochs
EARLY_STOP_PATIENCE=5     # stop after N evals without mIoU improvement

# GPU selection
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset ${DATASET} \
    --root_path ${ROOT_PATH} \
    --list_dir ${LIST_DIR} \
    --camus_split ${CAMUS_SPLIT} \
    --vit_name ${VIT_NAME} \
    --vit_patches_size ${VIT_PATCHES_SIZE} \
    --n_skip ${N_SKIP} \
    --img_size ${IMG_SIZE} \
    --max_iterations ${MAX_ITERATIONS} \
    --max_epochs ${MAX_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --base_lr ${BASE_LR} \
    --n_gpu ${N_GPU} \
    --deterministic ${DETERMINISTIC} \
    --seed ${SEED} \
    --eval_interval ${EVAL_INTERVAL} \
    --early_stop_patience ${EARLY_STOP_PATIENCE}