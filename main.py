"""
Main execution script for XAI-Enhanced GAN for Dorsal Hand Vein Authentication
Enhanced version with comprehensive evaluation and XAI analysis
"""
import argparse
import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm

from config import config
from data.data_loader import create_data_loaders
from models.gan import VeinGAN
from models.classifier import VeinAuthenticationClassifier
from explainability.gradcam import GradCAM
from explainability.shap_analysis import VeinSHAPAnalyzer
from explainability.lime_analysis import VeinLIMEAnalyzer
from evaluation.metrics import VeinMetrics
from evaluation.fairness import VeinFairnessAnalyzer
from training.train import VeinGANTrainer
from visualization.visualize import plot_vein_batch, compare_real_fake


class XAIVeinGANPipeline:
    """
    Complete pipeline for XAI-Enhanced Vein GAN
    Handles training, evaluation, generation, and explainability analysis
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self._setup_directories()
        
        # Initialize data loaders
        print("Loading datasets...")
        self.train_loader, self.val_loader, self.test_loader = self._load_data()
        
        # Initialize models
        print("Initializing models...")
        self.gan = VeinGAN().to(self.device)
        
        # Determine number of unique persons for classifier
        self.num_persons = self._get_num_unique_persons()
        self.classifier = VeinAuthenticationClassifier(
            num_classes=self.num_persons
        ).to(self.device)
        
        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            self._load_checkpoint(args.checkpoint)
        
        print(f"\n{'='*60}")
        print(f"XAI-Enhanced Vein GAN Pipeline Initialized")
        print(f"Device: {self.device}")
        print(f"Mode: {args.mode}")
        print(f"{'='*60}\n")
    
    def _setup_directories(self):
        """Create necessary output directories"""
        dirs = [
            config.RESULTS_DIR / 'generated',
            config.RESULTS_DIR / 'xai' / 'gradcam',
            config.RESULTS_DIR / 'xai' / 'shap',
            config.RESULTS_DIR / 'xai' / 'lime',
            config.RESULTS_DIR / 'metrics',
            config.RESULTS_DIR / 'fairness',
            config.RESULTS_DIR / 'comparisons',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _load_data(self) -> Tuple:
        """Load train, validation, and test data loaders"""
        db_paths = []
        if config.DB1_PATH.exists():
            db_paths.append(config.DB1_PATH)
        if config.DB2_PATH.exists():
            db_paths.append(config.DB2_PATH)
        
        if not db_paths:
            raise FileNotFoundError(
                f"No dataset found! Please ensure data is in:\n"
                f"  - {config.DB1_PATH}\n"
                f"  - {config.DB2_PATH}"
            )
        
        return create_data_loaders(db_paths)
    
    def _get_num_unique_persons(self) -> int:
        """Get number of unique persons in dataset"""
        dataset = self.train_loader.dataset.dataset
        person_ids = set(d['person_id'] for d in dataset.image_data)
        return len(person_ids)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        try:
            self.gan.load_checkpoint(checkpoint_path)
            print(f"✓ Loaded GAN checkpoint from {checkpoint_path}")
            
            # Try to load classifier checkpoint if it exists
            classifier_path = Path(checkpoint_path).parent / 'classifier_best.pth'
            if classifier_path.exists():
                state = torch.load(classifier_path, map_location=self.device)
                self.classifier.load_state_dict(state['model_state_dict'])
                print(f"✓ Loaded Classifier checkpoint from {classifier_path}")
        except Exception as e:
            print(f"⚠ Warning: Error loading checkpoint: {e}")
    
    def run(self):
        """Run the pipeline based on specified mode"""
        if self.args.mode == 'train':
            self.train_mode()
        elif self.args.mode == 'evaluate':
            self.evaluate_mode()
        elif self.args.mode == 'generate':
            self.generate_mode()
        elif self.args.mode == 'xai':
            self.xai_mode()
        else:
            raise ValueError(f"Unknown mode: {self.args.mode}")
    
    def train_mode(self):
        """Training mode - trains the GAN"""
        print("\n" + "="*60)
        print("TRAINING MODE")
        print("="*60 + "\n")
        
        trainer = VeinGANTrainer(
            self.gan,
            self.train_loader,
            self.val_loader
        )
        
        print(f"Starting training for {config.NUM_EPOCHS} epochs...")
        trainer.train()
        
        print("\n✓ Training completed!")
        print(f"Checkpoints saved in: {config.CHECKPOINT_DIR}")
    
    def evaluate_mode(self):
        """Comprehensive evaluation mode"""
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60 + "\n")
        
        self.gan.generator.eval()
        self.gan.discriminator.eval()
        self.classifier.eval()
        
        # 1. Quality Metrics
        print("1. Evaluating Image Quality Metrics...")
        quality_results = self._evaluate_quality_metrics()
        self._save_results(quality_results, 'quality_metrics.txt')
        
        # 2. Authentication Performance
        print("\n2. Evaluating Authentication Performance...")
        auth_results = self._evaluate_authentication()
        self._save_results(auth_results, 'authentication_metrics.txt')
        
        # 3. Fairness Analysis
        print("\n3. Performing Fairness Analysis...")
        fairness_results = self._evaluate_fairness()
        self._save_results(fairness_results, 'fairness_analysis.txt')
        
        # 4. Visual Comparison
        print("\n4. Generating Visual Comparisons...")
        self._generate_comparisons()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"\n✓ All evaluation results saved in: {config.RESULTS_DIR}")
        self._print_summary(quality_results, auth_results, fairness_results)
    
    def _evaluate_quality_metrics(self) -> Dict:
        """Evaluate image quality metrics (FID, IS, SSIM, PSNR)"""
        metrics_calc = VeinMetrics(device=self.device)
        results = {
            'ssim_scores': [],
            'psnr_scores': [],
        }
        
        with torch.no_grad():
            for i, (real_images, _) in enumerate(tqdm(self.test_loader, desc="Quality Metrics")):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                
                # Generate fake images
                fake_images = self.gan.generate(num_samples=batch_size)
                
                # Calculate metrics
                ssim_val, psnr_val = metrics_calc.calculate_ssim_psnr(
                    real_images, fake_images
                )
                
                results['ssim_scores'].append(ssim_val)
                results['psnr_scores'].append(psnr_val)
                
                # Limit evaluation to reasonable number of batches
                if i >= 20:
                    break
        
        # Aggregate results
        results['avg_ssim'] = np.mean(results['ssim_scores'])
        results['std_ssim'] = np.std(results['ssim_scores'])
        results['avg_psnr'] = np.mean(results['psnr_scores'])
        results['std_psnr'] = np.std(results['psnr_scores'])
        
        return results
    
    def _evaluate_authentication(self) -> Dict:
        """Evaluate authentication performance"""
        metrics_calc = VeinMetrics(device=self.device)
        results = {
            'predictions': [],
            'labels': [],
            'correct': 0,
            'total': 0
        }
        
        with torch.no_grad():
            for images, metadata in tqdm(self.test_loader, desc="Authentication"):
                images = images.to(self.device)
                
                # Get predictions
                outputs = self.classifier(images)
                predictions = torch.argmax(outputs, dim=1)
                labels = torch.tensor(
                    [metadata['person_id'][i] for i in range(len(metadata['person_id']))],
                    device=self.device
                )
                
                # Calculate accuracy
                results['correct'] += (predictions == labels).sum().item()
                results['total'] += labels.size(0)
                
                results['predictions'].extend(predictions.cpu().numpy())
                results['labels'].extend(labels.cpu().numpy())
        
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
        
        # Calculate EER if we have binary verification data
        # Note: This would require pair-wise verification setup
        
        return results
    
    def _evaluate_fairness(self) -> Dict:
        """Evaluate fairness across demographic groups"""
        fairness_analyzer = VeinFairnessAnalyzer(self.classifier)
        
        group_results, fairness_metrics = fairness_analyzer.evaluate_fairness(
            self.test_loader,
            device=self.device
        )
        
        results = {
            'group_performance': group_results,
            'fairness_metrics': fairness_metrics
        }
        
        return results
    
    def _generate_comparisons(self):
        """Generate visual comparisons between real and fake images"""
        # Get a batch of real images
        real_images, _ = next(iter(self.test_loader))
        real_images = real_images.to(self.device)
        
        # Generate fake images
        with torch.no_grad():
            fake_images = self.gan.generate(num_samples=real_images.size(0))
        
        # Save comparison
        save_path = config.RESULTS_DIR / 'comparisons' / 'real_vs_fake.png'
        compare_real_fake(real_images[:16], fake_images[:16], save_path=save_path)
        
        print(f"✓ Visual comparison saved: {save_path}")
    
    def generate_mode(self):
        """Generation mode - generates synthetic vein images"""
        print("\n" + "="*60)
        print("GENERATION MODE")
        print("="*60 + "\n")
        
        num_samples = self.args.num_samples
        print(f"Generating {num_samples} synthetic vein images...")
        
        self.gan.generator.eval()
        
        with torch.no_grad():
            # Generate in batches to avoid memory issues
            batch_size = 32
            all_images = []
            
            for i in tqdm(range(0, num_samples, batch_size)):
                current_batch_size = min(batch_size, num_samples - i)
                fake_images = self.gan.generate(num_samples=current_batch_size)
                all_images.append(fake_images)
            
            all_images = torch.cat(all_images, dim=0)
        
        # Save images
        output_path = config.RESULTS_DIR / 'generated' / f'synthetic_samples_{num_samples}.png'
        save_image(
            (all_images + 1) / 2,  # Denormalize
            output_path,
            nrow=int(np.sqrt(num_samples)),
            padding=2
        )
        
        print(f"\n✓ Generated {num_samples} images")
        print(f"✓ Saved to: {output_path}")
        
        # Save individual images if requested
        if self.args.save_individual:
            print("\nSaving individual images...")
            for i, img in enumerate(tqdm(all_images)):
                img_path = config.RESULTS_DIR / 'generated' / f'sample_{i:04d}.png'
                save_image((img + 1) / 2, img_path)
    
    def xai_mode(self):
        """Explainability mode - generates XAI visualizations"""
        print("\n" + "="*60)
        print("EXPLAINABILITY (XAI) MODE")
        print("="*60 + "\n")
        
        self.classifier.eval()
        
        # Get sample images
        images, metadata = next(iter(self.test_loader))
        images = images.to(self.device)
        num_samples = min(self.args.num_samples, len(images))
        
        # 1. Grad-CAM Analysis
        print("1. Running Grad-CAM Analysis...")
        self._run_gradcam(images[:num_samples], num_samples)
        
        # 2. SHAP Analysis
        print("\n2. Running SHAP Analysis...")
        self._run_shap(images, num_samples)
        
        # 3. LIME Analysis
        print("\n3. Running LIME Analysis...")
        self._run_lime(images[:num_samples], num_samples)
        
        print("\n" + "="*60)
        print("✓ XAI analysis completed!")
        print(f"✓ Results saved in: {config.RESULTS_DIR / 'xai'}")
        print("="*60)
    
    def _run_gradcam(self, images: torch.Tensor, num_samples: int):
        """Run Grad-CAM analysis"""
        gradcam = GradCAM(self.classifier, target_layer=config.GRADCAM_TARGET_LAYER)
        
        for i in tqdm(range(num_samples), desc="Grad-CAM"):
            try:
                cam, output, pred = gradcam.generate_cam(images[i:i+1])
                save_path = config.RESULTS_DIR / 'xai' / 'gradcam' / f'gradcam_{i:03d}.png'
                gradcam.visualize(images[i:i+1], cam, save_path=save_path)
                plt.close()  # Close to avoid memory leaks
            except Exception as e:
                print(f"  ⚠ Warning: Grad-CAM failed for sample {i}: {e}")
    
    def _run_shap(self, images: torch.Tensor, num_samples: int):
        """Run SHAP analysis"""
        try:
            background = images[:min(config.SHAP_BACKGROUND_SIZE, len(images))]
            shap_analyzer = VeinSHAPAnalyzer(
                self.classifier,
                background,
                device=self.device
            )
            shap_analyzer.create_explainer(method='gradient')
            
            for i in tqdm(range(num_samples), desc="SHAP"):
                try:
                    shap_vals, pred = shap_analyzer.explain_instance(images[i:i+1])
                    save_path = config.RESULTS_DIR / 'xai' / 'shap' / f'shap_{i:03d}.png'
                    shap_analyzer.visualize_shap_instance(
                        images[i:i+1],
                        shap_vals,
                        save_path=save_path
                    )
                    plt.close()
                except Exception as e:
                    print(f"  ⚠ Warning: SHAP failed for sample {i}: {e}")
        except Exception as e:
            print(f"  ⚠ Warning: SHAP initialization failed: {e}")
    
    def _run_lime(self, images: torch.Tensor, num_samples: int):
        """Run LIME analysis"""
        try:
            lime_analyzer = VeinLIMEAnalyzer(self.classifier, device=self.device)
            
            for i in tqdm(range(num_samples), desc="LIME"):
                try:
                    exp, pred, _ = lime_analyzer.explain_instance(
                        images[i:i+1],
                        num_samples=config.LIME_NUM_SAMPLES
                    )
                    save_path = config.RESULTS_DIR / 'xai' / 'lime' / f'lime_{i:03d}.png'
                    lime_analyzer.visualize_explanation(
                        images[i:i+1],
                        exp,
                        pred,
                        save_path=save_path
                    )
                    plt.close()
                except Exception as e:
                    print(f"  ⚠ Warning: LIME failed for sample {i}: {e}")
        except Exception as e:
            print(f"  ⚠ Warning: LIME initialization failed: {e}")
    
    def _save_results(self, results: Dict, filename: str):
        """Save evaluation results to file"""
        output_path = config.RESULTS_DIR / 'metrics' / filename
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"Results: {filename}\n")
            f.write("="*60 + "\n\n")
            
            for key, value in results.items():
                if isinstance(value, (list, np.ndarray)):
                    f.write(f"{key}: [array of {len(value)} items]\n")
                elif isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"  ✓ Saved: {output_path}")
    
    def _print_summary(self, quality_results: Dict, auth_results: Dict, fairness_results: Dict):
        """Print evaluation summary"""
        print(f"\nQuality Metrics:")
        print(f"  SSIM: {quality_results['avg_ssim']:.4f} ± {quality_results['std_ssim']:.4f}")
        print(f"  PSNR: {quality_results['avg_psnr']:.2f} ± {quality_results['std_psnr']:.2f} dB")
        
        print(f"\nAuthentication Performance:")
        print(f"  Accuracy: {auth_results['accuracy']:.4f} ({auth_results['correct']}/{auth_results['total']})")
        
        print(f"\nFairness Analysis:")
        for group, metrics in fairness_results['group_performance'].items():
            print(f"  {group}: Accuracy = {metrics['accuracy']:.4f}, Samples = {metrics['sample_size']}")
        
        if fairness_results['fairness_metrics']:
            print(f"\n  Fairness Metrics:")
            for metric, value in fairness_results['fairness_metrics'].items():
                print(f"    {metric}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="XAI-Enhanced GAN for Dorsal Hand Vein Authentication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py --mode train --epochs 200

  # Evaluate with XAI
  python main.py --mode evaluate --checkpoint checkpoints/best_model.pth

  # Generate synthetic samples
  python main.py --mode generate --num_samples 100 --save_individual

  # Run XAI analysis only
  python main.py --mode xai --checkpoint checkpoints/best_model.pth --num_samples 20
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="evaluate",
        choices=["train", "evaluate", "generate", "xai"],
        help="Mode of operation"
    )
    
    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples for generation/analysis"
    )
    
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual generated images"
    )
    
    # Training parameters (override config)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--eval_metrics",
        nargs='+',
        default=None,
        help="Specific metrics to evaluate"
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    
    try:
        # Initialize and run pipeline
        pipeline = XAIVeinGANPipeline(args)
        pipeline.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()