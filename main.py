"""
Main execution script for XAI-Enhanced GAN for Dorsal Hand Vein Authentication
"""
import argparse
import torch
import numpy as np
import os
from pathlib import Path

from config import config
from data.data_loader import create_data_loaders
from models.gan import VeinGAN
from models.classifier import VeinAuthenticationClassifier
from explainability.gradcam import GradCAM
from explainability.shap_analysis import VeinSHAPAnalyzer
from explainability.lime_analysis import VeinLIMEAnalyzer
# Inside main.py --mode evaluate
from evaluation.metrics import VeinMetrics
from evaluation.fairness import VeinFairnessAnalyzer

def main():
    parser = argparse.ArgumentParser(description="XAI-Enhanced Vein GAN")
    parser.add_argument("--mode", type=str, default="evaluate", choices=["train", "evaluate", "generate"],
                        help="Mode of operation: train, evaluate, or generate")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples for generation/analysis")
    args = parser.parse_args()

    # Create data loaders
    db_paths = [config.DB1_PATH, config.DB2_PATH]
    train_loader, val_loader, test_loader = create_data_loaders(db_paths)

    # Initialize GAN
    gan = VeinGAN()
    
    # Initialize Classifier (using 100 classes as a default placeholder)
    classifier = VeinAuthenticationClassifier(num_classes=100).to(gan.device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        gan.load_checkpoint(args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")

    if args.mode == "train":
        print("Starting training mode... (Training logic would be called from training/train.py)")
        # Example: trainer.train(train_loader, val_loader)
        
    elif args.mode == "evaluate":
        print("Starting XAI-Enhanced Evaluation...")
        # Get a sample batch for analysis
        images, metadata = next(iter(test_loader))
        images = images.to(gan.device)
        
        # 1. Grad-CAM Analysis
        print("Running Grad-CAM...")
        gradcam = GradCAM(classifier, target_layer=config.GRADCAM_TARGET_LAYER)
        for i in range(min(args.num_samples, len(images))):
            cam, _, pred = gradcam.generate_cam(images[i:i+1])
            gradcam.visualize(images[i:i+1], cam, save_path=config.RESULTS_DIR / f"gradcam_{i}.png")

        # 2. SHAP Analysis
        print("Running SHAP...")
        background = images[:config.SHAP_BACKGROUND_SIZE]
        shap_analyzer = VeinSHAPAnalyzer(classifier, background, device=gan.device)
        shap_analyzer.create_explainer(method='gradient')
        for i in range(min(args.num_samples, len(images))):
            shap_vals, pred = shap_analyzer.explain_instance(images[i:i+1])
            shap_analyzer.visualize_shap_instance(images[i:i+1], shap_vals, 
                                                 save_path=config.RESULTS_DIR / f"shap_{i}.png")

        # 3. LIME Analysis
        print("Running LIME...")
        lime_analyzer = VeinLIMEAnalyzer(classifier, device=gan.device)
        for i in range(min(args.num_samples, len(images))):
            exp, pred, _ = lime_analyzer.explain_instance(images[i:i+1], num_samples=config.LIME_NUM_SAMPLES)
            lime_analyzer.visualize_explanation(images[i:i+1], exp, pred, 
                                               save_path=config.RESULTS_DIR / f"lime_{i}.png")

        # 1. Quality Metrics
        metrics_calc = VeinMetrics()
        avg_ssim, avg_psnr = metrics_calc.calculate_ssim_psnr(real_batch, fake_batch)
        print(f"Batch SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f}")

        # 2. Fairness Check
        fairness_checker = VeinFairnessAnalyzer(classifier)
        group_results, fairness_metrics = fairness_checker.evaluate_fairness(test_loader)
        print(f"Fairness Metrics: {fairness_metrics}")

    elif args.mode == "generate":
        print(f"Generating {args.num_samples} synthetic vein images...")
        fake_images = gan.generate(num_samples=args.num_samples)
        from torchvision.utils import save_image
        save_image((fake_images + 1) / 2, config.RESULTS_DIR / "generated_samples.png", nrow=5)
        print(f"Samples saved to {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()