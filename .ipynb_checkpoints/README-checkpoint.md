# XAI-Enhanced GAN for Dorsal Hand Vein Biometric Authentication

## Project Overview
Implementation of "XAI-Enhanced Generative Adversarial Networks for Explainable Dorsal Hand Vein Biometric Authentication"

## Dataset Information
- **Database 1**: 138 people, 4 images per hand (1,104 images total)
- **Database 2**: 113 people, 3 images per hand (678 images total)
- **Image Resolution**: 752 × 560 pixels
- **Format**: 16-bit TIFF (.tif)
- **Naming Convention**: `person_[xxx]_db[1|2]_[L|R][1-4].tif`

## Key Features
- **GAN Architecture**: Deep learning-based generative model for vein pattern synthesis
- **Explainability**: Integration of Grad-CAM, SHAP, and LIME
- **Fairness Analysis**: Demographic bias detection and mitigation
- **Attention Mechanisms**: Enhanced generator for better synthetic patterns

## Project Structure
```
xai_vein_gan/
├── data/
│   ├── data_loader.py          # Dataset loading and preprocessing
│   └── augmentation.py         # Data augmentation utilities
├── models/
│   ├── generator.py            # GAN Generator with Attention
│   ├── discriminator.py        # GAN Discriminator
│   └── gan.py                  # Combined GAN model
├── explainability/
│   ├── gradcam.py             # Grad-CAM implementation
│   ├── shap_explainer.py      # SHAP integration
│   └── lime_explainer.py      # LIME integration
├── training/
│   ├── train.py               # Training loop
│   └── losses.py              # Custom loss functions
├── evaluation/
│   ├── metrics.py             # Evaluation metrics
│   └── fairness.py            # Bias detection and analysis
├── visualization/
│   └── visualize.py           # Visualization utilities
├── config.py                   # Configuration settings
└── main.py                     # Main execution script
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
```python
python prepare_data.py --db1_path /path/to/DorsalHandVeins_DB1 \
                       --db2_path /path/to/DorsalHandVeins_DB2
```

### 2. Training
```python
python main.py --mode train --epochs 200 --batch_size 32
```

### 3. Evaluation with XAI
```python
python main.py --mode evaluate --checkpoint ./checkpoints/best_model.pth
```

### 4. Generate Synthetic Samples
```python
python main.py --mode generate --num_samples 100
```

## Configuration
Edit `config.py` to customize:
- Model architecture parameters
- Training hyperparameters
- XAI method settings
- Data paths and preprocessing options

## Results
Results will be saved in:
- `./results/generated/` - Synthetic vein images
- `./results/xai/` - Explainability visualizations
- `./results/metrics/` - Performance metrics
- `./results/fairness/` - Bias analysis reports

## Citation
```bibtex
@article{xai_vein_gan_2025,
  title={XAI-Enhanced Generative Adversarial Networks for Explainable Dorsal Hand Vein Biometric Authentication},
  author={Your Name},
  year={2025}
}
```

## License
MIT License