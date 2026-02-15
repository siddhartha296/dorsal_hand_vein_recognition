"""
Comprehensive Training Loop for XAI-Enhanced Vein GAN
Includes advanced training strategies, logging, and checkpointing
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import time
import json
from collections import defaultdict

from training.losses import CombinedGANLoss
from config import config
from visualization.visualize import plot_vein_batch, compare_real_fake


class VeinGANTrainer:
    """
    Comprehensive trainer for Vein GAN with advanced features
    """
    
    def __init__(
        self,
        gan,
        train_loader,
        val_loader,
        device: str = None,
        use_mixed_precision: bool = True,
        use_tensorboard: bool = True
    ):
        """
        Args:
            gan: VeinGAN model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            use_mixed_precision: Use automatic mixed precision
            use_tensorboard: Enable TensorBoard logging
        """
        self.gan = gan
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or config.DEVICE
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Optimizers
        self.optimizer_G = self.gan.optimizer_g
        self.optimizer_D = self.gan.optimizer_d
        
        # Learning rate schedulers
        self.scheduler_G = self._create_scheduler(self.optimizer_G)
        self.scheduler_D = self._create_scheduler(self.optimizer_D)
        
        # Loss calculator
        self.loss_calculator = CombinedGANLoss(
            device=self.device,
            loss_type=getattr(config, 'GAN_LOSS_TYPE', 'vanilla')
        )
        
        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision and config.MIXED_PRECISION
        self.scaler_G = GradScaler() if self.use_mixed_precision else None
        self.scaler_D = GradScaler() if self.use_mixed_precision else None
        
        # Logging
        self.use_tensorboard = use_tensorboard and config.USE_TENSORBOARD
        self.writer = SummaryWriter(log_dir=config.LOGS_DIR) if self.use_tensorboard else None
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Early stopping
        self.patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 20)
        self.patience_counter = 0
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mixed Precision: {self.use_mixed_precision}")
        print(f"  TensorBoard: {self.use_tensorboard}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        scheduler_type = getattr(config, 'LR_SCHEDULER', 'step')
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(config, 'LR_DECAY_STEP', 50),
                gamma=getattr(config, 'LR_DECAY_GAMMA', 0.5)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.NUM_EPOCHS,
                eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train (overrides config)
        """
        num_epochs = num_epochs or config.NUM_EPOCHS
        
        print(f"\n{'='*70}")
        print(f"Starting Training for {num_epochs} epochs")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            if epoch % config.LOG_INTERVAL == 0:
                val_metrics = self.validate()
                
                # Log metrics
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # Check for improvement
                current_metric = val_metrics.get('g_loss', float('inf'))
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self._save_checkpoint('best_model.pth', is_best=True)
                    print(f"  ✓ New best model saved (metric: {current_metric:.4f})")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Update learning rates
            self._update_learning_rates(val_metrics if epoch % config.LOG_INTERVAL == 0 else None)
            
            # Generate visualizations
            if (epoch + 1) % config.VISUALIZE_INTERVAL == 0:
                self._generate_visualizations(epoch)
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best metric: {self.best_metric:.4f}")
        print(f"{'='*70}\n")
        
        # Save final model
        self._save_checkpoint('final_model.pth')
        
        # Close writer
        if self.writer:
            self.writer.close()
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of epoch metrics
        """
        self.gan.generator.train()
        self.gan.discriminator.train()
        
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{config.NUM_EPOCHS}",
            leave=True
        )
        
        for batch_idx, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # ==================== Train Discriminator ====================
            d_metrics = self._train_discriminator_step(real_images)
            
            # ==================== Train Generator ====================
            # Train generator every N_CRITIC steps
            if batch_idx % config.N_CRITIC == 0:
                g_metrics = self._train_generator_step(real_images)
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f"{d_metrics['total']:.4f}",
                    'G_loss': f"{g_metrics['total']:.4f}"
                })
                
                # Accumulate metrics
                for key, value in g_metrics.items():
                    epoch_metrics[f'g_{key}'].append(value)
            
            # Accumulate discriminator metrics
            for key, value in d_metrics.items():
                epoch_metrics[f'd_{key}'].append(value)
            
            self.global_step += 1
            
            # Log to TensorBoard
            if self.writer and batch_idx % config.LOG_INTERVAL == 0:
                self._log_batch_metrics(d_metrics, g_metrics if batch_idx % config.N_CRITIC == 0 else None)
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return avg_metrics
    
    def _train_discriminator_step(self, real_images: torch.Tensor) -> Dict[str, float]:
        """Train discriminator for one step"""
        batch_size = real_images.size(0)
        
        # Zero gradients
        self.optimizer_D.zero_grad()
        
        # Mixed precision context
        with autocast(enabled=self.use_mixed_precision):
            # Generate fake images
            z = torch.randn(batch_size, config.LATENT_DIM).to(self.device)
            fake_images = self.gan.generator(z)
            
            # Discriminator predictions (with features for feature matching)
            real_pred, real_features = self.gan.discriminator(real_images, return_features=True)
            fake_pred, fake_features = self.gan.discriminator(fake_images.detach(), return_features=True)
            
            # Compute discriminator loss
            d_loss, d_metrics = self.loss_calculator.compute_discriminator_loss(
                real_pred,
                fake_pred,
                discriminator=self.gan.discriminator,
                real_samples=real_images,
                fake_samples=fake_images.detach()
            )
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler_D.scale(d_loss).backward()
            self.scaler_D.step(self.optimizer_D)
            self.scaler_D.update()
        else:
            d_loss.backward()
            self.optimizer_D.step()
        
        return d_metrics
    
    def _train_generator_step(self, real_images: torch.Tensor) -> Dict[str, float]:
        """Train generator for one step"""
        batch_size = real_images.size(0)
        
        # Zero gradients
        self.optimizer_G.zero_grad()
        
        # Mixed precision context
        with autocast(enabled=self.use_mixed_precision):
            # Generate fake images
            z = torch.randn(batch_size, config.LATENT_DIM).to(self.device)
            fake_images = self.gan.generator(z)
            
            # Discriminator predictions with features
            fake_pred, fake_features = self.gan.discriminator(fake_images, return_features=True)
            real_pred, real_features = self.gan.discriminator(real_images, return_features=True)
            
            # Compute generator loss
            g_loss, g_metrics = self.loss_calculator.compute_generator_loss(
                fake_pred,
                real_images,
                fake_images,
                real_features,
                fake_features
            )
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler_G.scale(g_loss).backward()
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()
        else:
            g_loss.backward()
            self.optimizer_G.step()
        
        return g_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary of validation metrics
        """
        self.gan.generator.eval()
        self.gan.discriminator.eval()
        
        val_metrics = defaultdict(list)
        
        for real_images, _ in self.val_loader:
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Generate fake images
            z = torch.randn(batch_size, config.LATENT_DIM).to(self.device)
            fake_images = self.gan.generator(z)
            
            # Get predictions
            real_pred, real_features = self.gan.discriminator(real_images, return_features=True)
            fake_pred, fake_features = self.gan.discriminator(fake_images, return_features=True)
            
            # Compute losses
            d_loss, d_dict = self.loss_calculator.compute_discriminator_loss(
                real_pred, fake_pred
            )
            g_loss, g_dict = self.loss_calculator.compute_generator_loss(
                fake_pred, real_images, fake_images, real_features, fake_features
            )
            
            # Accumulate metrics
            for key, value in d_dict.items():
                val_metrics[f'd_{key}'].append(value)
            for key, value in g_dict.items():
                val_metrics[f'g_{key}'].append(value)
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        return avg_metrics
    
    def _update_learning_rates(self, val_metrics: Optional[Dict] = None):
        """Update learning rate schedulers"""
        if self.scheduler_G is not None:
            if isinstance(self.scheduler_G, optim.lr_scheduler.ReduceLROnPlateau):
                if val_metrics is not None:
                    self.scheduler_G.step(val_metrics.get('g_loss', 0))
            else:
                self.scheduler_G.step()
        
        if self.scheduler_D is not None:
            if isinstance(self.scheduler_D, optim.lr_scheduler.ReduceLROnPlateau):
                if val_metrics is not None:
                    self.scheduler_D.step(val_metrics.get('d_loss', 0))
            else:
                self.scheduler_D.step()
    
    def _log_batch_metrics(
        self,
        d_metrics: Dict[str, float],
        g_metrics: Optional[Dict[str, float]] = None
    ):
        """Log batch metrics to TensorBoard"""
        if not self.writer:
            return
        
        # Discriminator metrics
        for key, value in d_metrics.items():
            self.writer.add_scalar(f'Train_Batch/D_{key}', value, self.global_step)
        
        # Generator metrics
        if g_metrics:
            for key, value in g_metrics.items():
                self.writer.add_scalar(f'Train_Batch/G_{key}', value, self.global_step)
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch metrics to console and TensorBoard"""
        # Print to console
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"  Train - D_loss: {train_metrics['d_total']:.4f}, G_loss: {train_metrics['g_total']:.4f}")
        print(f"  Val   - D_loss: {val_metrics['d_total']:.4f}, G_loss: {val_metrics['g_total']:.4f}")
        
        # Log to TensorBoard
        if self.writer:
            # Training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train_Epoch/{key}', value, epoch)
            
            # Validation metrics
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val_Epoch/{key}', value, epoch)
            
            # Learning rates
            self.writer.add_scalar('LR/generator', self.optimizer_G.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('LR/discriminator', self.optimizer_D.param_groups[0]['lr'], epoch)
        
        # Store in history
        for key, value in train_metrics.items():
            self.train_metrics[key].append(value)
        for key, value in val_metrics.items():
            self.val_metrics[key].append(value)
    
    @torch.no_grad()
    def _generate_visualizations(self, epoch: int):
        """Generate and save visualizations"""
        self.gan.generator.eval()
        
        # Generate samples
        num_samples = 16
        z = torch.randn(num_samples, config.LATENT_DIM).to(self.device)
        fake_images = self.gan.generator(z)
        
        # Save generated samples
        save_path = config.RESULTS_DIR / 'training_progress' / f'epoch_{epoch+1}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot_vein_batch(fake_images, title=f"Generated Samples - Epoch {epoch+1}", save_path=save_path)
        
        # Log to TensorBoard
        if self.writer:
            from torchvision.utils import make_grid
            grid = make_grid((fake_images + 1) / 2, nrow=4, padding=2, normalize=False)
            self.writer.add_image('Generated_Samples', grid, epoch)
        
        # Compare with real images
        real_images, _ = next(iter(self.val_loader))
        real_images = real_images[:num_samples].to(self.device)
        
        comparison_path = config.RESULTS_DIR / 'training_progress' / f'comparison_epoch_{epoch+1}.png'
        compare_real_fake(real_images, fake_images, save_path=comparison_path)
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint_path = config.CHECKPOINT_DIR / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.gan.generator.state_dict(),
            'discriminator_state_dict': self.gan.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_G.state_dict(),
            'optimizer_d_state_dict': self.optimizer_D.state_dict(),
            'best_metric': self.best_metric,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'config': {
                'latent_dim': config.LATENT_DIM,
                'image_size': config.IMAGE_SIZE,
                'batch_size': config.BATCH_SIZE,
            }
        }
        
        # Add scheduler states if they exist
        if self.scheduler_G is not None:
            checkpoint['scheduler_g_state_dict'] = self.scheduler_G.state_dict()
        if self.scheduler_D is not None:
            checkpoint['scheduler_d_state_dict'] = self.scheduler_D.state_dict()
        
        # Add scaler states for mixed precision
        if self.scaler_G is not None:
            checkpoint['scaler_g_state_dict'] = self.scaler_G.state_dict()
        if self.scaler_D is not None:
            checkpoint['scaler_d_state_dict'] = self.scaler_D.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save training history as JSON
        history_path = config.CHECKPOINT_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train_metrics': {k: [float(v) for v in vals] for k, vals in self.train_metrics.items()},
                'val_metrics': {k: [float(v) for v in vals] for k, vals in self.val_metrics.items()}
            }, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizer states
        self.optimizer_G.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        # Load metrics history
        if 'train_metrics' in checkpoint:
            self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
        if 'val_metrics' in checkpoint:
            self.val_metrics = defaultdict(list, checkpoint['val_metrics'])
        
        # Load scheduler states
        if self.scheduler_G and 'scheduler_g_state_dict' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_g_state_dict'])
        if self.scheduler_D and 'scheduler_d_state_dict' in checkpoint:
            self.scheduler_D.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        # Load scaler states
        if self.scaler_G and 'scaler_g_state_dict' in checkpoint:
            self.scaler_G.load_state_dict(checkpoint['scaler_g_state_dict'])
        if self.scaler_D and 'scaler_d_state_dict' in checkpoint:
            self.scaler_D.load_state_dict(checkpoint['scaler_d_state_dict'])
        
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best metric: {self.best_metric:.4f}")


def resume_training(checkpoint_path: str, gan, train_loader, val_loader):
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        gan: VeinGAN model
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    trainer = VeinGANTrainer(gan, train_loader, val_loader)
    trainer.load_checkpoint(checkpoint_path)
    trainer.train()


if __name__ == "__main__":
    # Test trainer
    print("Testing VeinGANTrainer...")
    
    from models.gan import VeinGAN
    from data.data_loader import create_data_loaders
    
    # Create dummy data loaders
    print("Creating data loaders...")
    db_paths = [config.DB1_PATH, config.DB2_PATH]
    train_loader, val_loader, test_loader = create_data_loaders(db_paths)
    
    # Create GAN
    print("Creating GAN...")
    gan = VeinGAN()
    
    # Create trainer
    print("Creating trainer...")
    trainer = VeinGANTrainer(gan, train_loader, val_loader)
    
    print("\n✓ Trainer initialized successfully!")
    print(f"  Ready to train for {config.NUM_EPOCHS} epochs")