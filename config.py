"""
Configuration file for XAI-Enhanced GAN for Dorsal Hand Vein Authentication
"""
import os
from pathlib import Path

class Config:
    # ==================== Paths ====================
    PROJECT_ROOT = Path(__file__).parent
    DATA_ROOT = PROJECT_ROOT / 'data'
    DB1_PATH = DATA_ROOT / 'DorsalHandVeins_DB1'
    DB2_PATH = DATA_ROOT / 'DorsalHandVeins_DB2'
    
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    # Create directories if they don't exist
    for dir_path in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== Data Parameters ====================
    IMAGE_SIZE = (256, 256)  # Resize to this for training
    ORIGINAL_SIZE = (752, 560)
    NUM_CHANNELS = 1  # Grayscale
    
    # Training/Validation split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_PARAMS = {
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'zoom_range': 0.1,
        'brightness_range': (0.8, 1.2)
    }
    
    # ==================== Model Architecture ====================
    # Generator
    LATENT_DIM = 128
    GENERATOR_FILTERS = [512, 256, 128, 64]
    USE_ATTENTION = True
    ATTENTION_HEADS = 8
    
    # Discriminator
    DISCRIMINATOR_FILTERS = [64, 128, 256, 512]
    DISCRIMINATOR_DROPOUT = 0.3
    SPECTRAL_NORM = True
    
    # ==================== Training Parameters ====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE_G = 0.0002
    LEARNING_RATE_D = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999
    
    # Loss weights
    ADVERSARIAL_LOSS_WEIGHT = 1.0
    FEATURE_MATCHING_WEIGHT = 10.0
    PERCEPTUAL_LOSS_WEIGHT = 10.0
    
    # Training strategy
    N_CRITIC = 5  # Train discriminator N times per generator update
    GRADIENT_PENALTY_WEIGHT = 10.0  # For WGAN-GP
    
    # ==================== XAI Parameters ====================
    # Grad-CAM
    GRADCAM_TARGET_LAYER = 'layer4'
    GRADCAM_COLORMAP = 'jet'
    
    # SHAP
    SHAP_NUM_SAMPLES = 100
    SHAP_BACKGROUND_SIZE = 50
    
    # LIME
    LIME_NUM_SAMPLES = 1000
    LIME_NUM_FEATURES = 100
    
    # ==================== Evaluation Parameters ====================
    EVALUATION_METRICS = [
        'FID',  # Frechet Inception Distance
        'IS',   # Inception Score
        'SSIM', # Structural Similarity
        'PSNR', # Peak Signal-to-Noise Ratio
        'EER',  # Equal Error Rate (for authentication)
    ]
    
    # Authentication thresholds
    AUTH_THRESHOLD = 0.5
    FAR_TARGET = 0.001  # False Acceptance Rate target
    
    # ==================== Fairness Analysis ====================
    FAIRNESS_ATTRIBUTES = ['hand_side']  # Left vs Right
    FAIRNESS_METRICS = [
        'demographic_parity',
        'equal_opportunity',
        'equalized_odds'
    ]
    
    # ==================== Hardware & Performance ====================
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    NUM_WORKERS = 4
    PIN_MEMORY = True
    MIXED_PRECISION = True  # Use automatic mixed precision
    
    # ==================== Logging & Checkpointing ====================
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    VISUALIZE_INTERVAL = 5  # Generate visualizations every N epochs
    
    USE_TENSORBOARD = True
    USE_WANDB = False  # Set to True if using Weights & Biases
    WANDB_PROJECT = 'xai-vein-gan'
    
    # ==================== Reproducibility ====================
    RANDOM_SEED = 42
    DETERMINISTIC = True


# Create a default config instance
config = Config()