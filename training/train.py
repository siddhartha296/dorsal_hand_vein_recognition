"""
Comprehensive Training Loop for XAI-Enhanced Vein GAN
Fixed version:
  - Single shared GradScaler (not two separate ones)
  - Fake images generated ONCE per iteration, reused for D and G steps
  - Gradient penalty computed OUTSIDE autocast in float32
  - g_metrics initialized before loop to avoid NameError
  - Gradient clipping on both G and D to prevent exploding gradients / NaN
  - NaN detection in generator output — skips G step if NaN found
  - Scheduler step order fixed (optimizer.step BEFORE scheduler.step)
  - Progressive loss weight warmup: style/perceptual weights increase after warmup_epochs
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
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


# Max gradient norm for clipping — prevents exploding gradients that cause NaN
GRAD_CLIP_NORM = 1.0

# After this many epochs, increase style/perceptual loss weights gradually
WARMUP_EPOCHS = 10


class VeinGANTrainer:
    """
    Comprehensive trainer for Vein GAN.

    Key fixes vs original:
    1. Single GradScaler shared by both G and D.
    2. Fake images generated ONCE per iteration; D gets .detach(), G reuses same tensor.
    3. Gradient penalty computed outside autocast.
    4. g_metrics = {} initialized before the batch loop.
    5. Gradient clipping (max_norm=1.0) on both G and D after unscale.
    6. NaN detection in fake_images before G step.
    7. Scheduler step called AFTER optimizer step (correct PyTorch order).
    8. Progressive loss weight warmup after WARMUP_EPOCHS.
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
        self.gan = gan
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or config.DEVICE

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

        self.optimizer_G = self.gan.optimizer_g
        self.optimizer_D = self.gan.optimizer_d

        # Loss calculator (starts with safe low weights for style/perceptual)
        self.loss_calculator = CombinedGANLoss(
            device=self.device,
            loss_type=getattr(config, 'GAN_LOSS_TYPE', 'vanilla')
        )

        # FIX 1: Single shared GradScaler
        self.use_mixed_precision = use_mixed_precision and config.MIXED_PRECISION
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None

        self.use_tensorboard = use_tensorboard and config.USE_TENSORBOARD
        self.writer = SummaryWriter(log_dir=config.LOGS_DIR) if self.use_tensorboard else None

        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        self.patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 20)
        self.patience_counter = 0

        # FIX 7: Create schedulers AFTER optimizer is ready
        # They will be stepped at end of epoch (after optimizer.step)
        self.scheduler_G = self._create_scheduler(self.optimizer_G)
        self.scheduler_D = self._create_scheduler(self.optimizer_D)

        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mixed Precision: {self.use_mixed_precision}")
        print(f"  TensorBoard: {self.use_tensorboard}")
        print(f"  Gradient clip norm: {GRAD_CLIP_NORM}")
        print(f"  Loss warmup epochs: {WARMUP_EPOCHS}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")

    def _create_scheduler(self, optimizer):
        scheduler_type = getattr(config, 'LR_SCHEDULER', 'step')

        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(config, 'LR_DECAY_STEP', 50),
                gamma=getattr(config, 'LR_DECAY_GAMMA', 0.5)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
        else:
            return None

    # ==================== Main Training Loop ====================

    def train(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or config.NUM_EPOCHS

        print(f"\n{'='*70}")
        print(f"Starting Training for {num_epochs} epochs")
        print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # FIX 8: Increase loss weights after warmup
            if epoch == WARMUP_EPOCHS:
                print(f"\n  Epoch {epoch+1}: Warmup complete — increasing style/perceptual weights")
                self.loss_calculator.update_weights({
                    'perceptual': 5.0,
                    'style':      10.0,
                })
            elif epoch == WARMUP_EPOCHS * 2:
                print(f"\n  Epoch {epoch+1}: Further increasing style/perceptual weights")
                self.loss_calculator.update_weights({
                    'perceptual': 10.0,
                    'style':      50.0,
                })

            train_metrics = self.train_epoch()

            # Validation and logging every LOG_INTERVAL epochs
            if epoch % config.LOG_INTERVAL == 0:
                val_metrics = self.validate()
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)

                current_metric = val_metrics.get('g_total', float('inf'))
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self._save_checkpoint('best_model.pth', is_best=True)
                    print(f"  ✓ New best model saved (metric: {current_metric:.4f})")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                val_metrics = {}

            # FIX 7: Scheduler stepped AFTER optimizer — correct PyTorch order
            self._update_learning_rates(val_metrics if val_metrics else None)

            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

            if (epoch + 1) % config.VISUALIZE_INTERVAL == 0:
                self._generate_visualizations(epoch)

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best metric: {self.best_metric:.4f}")
        print(f"{'='*70}\n")

        self._save_checkpoint('final_model.pth')

        if self.writer:
            self.writer.close()

    # ==================== Epoch ====================

    def train_epoch(self) -> Dict[str, float]:
        self.gan.generator.train()
        self.gan.discriminator.train()

        epoch_metrics = defaultdict(list)

        # FIX 4: Initialize g_metrics before loop — prevents NameError on
        # D-only batches (when batch_idx % N_CRITIC != 0)
        g_metrics = {}

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{config.NUM_EPOCHS}",
            leave=True
        )

        for batch_idx, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)

            # ------------------------------------------------------------------
            # FIX 2: Generate fake images ONCE per iteration.
            # D step uses fake_images.detach() — no generator gradients flow.
            # G step reuses the same fake_images tensor so feature matching
            # compares the correct real vs fake feature distributions.
            # ------------------------------------------------------------------
            z = torch.randn(batch_size, config.LATENT_DIM).to(self.device)

            with autocast('cuda', enabled=self.use_mixed_precision):
                fake_images = self.gan.generator(z)

            # FIX 6: Skip entire iteration if generator produced NaN
            if torch.isnan(fake_images).any() or torch.isinf(fake_images).any():
                print(f"\n⚠ NaN/Inf in generator output at step {self.global_step} — skipping batch")
                self.global_step += 1
                continue

            # ---- Train Discriminator ----
            d_metrics = self._train_discriminator_step(real_images, fake_images.detach())

            # ---- Train Generator every N_CRITIC steps ----
            if batch_idx % config.N_CRITIC == 0:
                g_metrics = self._train_generator_step(real_images, fake_images)

                pbar.set_postfix({
                    'D_loss': f"{d_metrics['total']:.4f}",
                    'G_loss': f"{g_metrics['total']:.4f}"
                })

                for key, value in g_metrics.items():
                    epoch_metrics[f'g_{key}'].append(value)

            for key, value in d_metrics.items():
                epoch_metrics[f'd_{key}'].append(value)

            self.global_step += 1

            if self.writer and batch_idx % config.LOG_INTERVAL == 0:
                self._log_batch_metrics(d_metrics, g_metrics)

        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        return avg_metrics

    # ==================== Discriminator Step ====================

    def _train_discriminator_step(
        self,
        real_images: torch.Tensor,
        fake_images_detached: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train discriminator for one step.

        FIX 3: Gradient penalty computed OUTSIDE autocast in float32.
        FIX 5: Gradient clipping applied after scaler.unscale_().
        """
        self.optimizer_D.zero_grad()

        # Forward pass under autocast
        with autocast('cuda', enabled=self.use_mixed_precision):
            real_pred, real_features = self.gan.discriminator(real_images, return_features=True)
            fake_pred, _ = self.gan.discriminator(fake_images_detached, return_features=True)

            d_loss, d_metrics = self.loss_calculator.compute_discriminator_loss(
                real_pred, fake_pred
            )

        # FIX 3: Gradient penalty OUTSIDE autocast — needs float32 tensors
        use_gp = getattr(config, 'GAN_LOSS_TYPE', 'vanilla') in ('wgan', 'wgan-gp')
        if use_gp:
            gp_loss = self.loss_calculator.losses.get_gradient_penalty(
                self.gan.discriminator,
                real_images.float(),
                fake_images_detached.float(),
                lambda_gp=config.GRADIENT_PENALTY_WEIGHT
            )
            d_metrics['gradient_penalty'] = gp_loss.item()
            d_metrics['total'] = d_metrics['total'] + gp_loss.item()
            d_loss = d_loss + gp_loss

        # Backward + FIX 5: clip gradients before optimizer step
        if self.use_mixed_precision:
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.optimizer_D)
            torch.nn.utils.clip_grad_norm_(
                self.gan.discriminator.parameters(), max_norm=GRAD_CLIP_NORM
            )
            self.scaler.step(self.optimizer_D)
            self.scaler.update()
        else:
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.gan.discriminator.parameters(), max_norm=GRAD_CLIP_NORM
            )
            self.optimizer_D.step()

        return d_metrics

    # ==================== Generator Step ====================

    def _train_generator_step(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train generator for one step.

        FIX 2: Reuses the same fake_images from train_epoch (with grads).
        FIX 5: Gradient clipping applied after scaler.unscale_().
        FIX 6: NaN guard before backward.
        """
        self.optimizer_G.zero_grad()

        with autocast('cuda', enabled=self.use_mixed_precision):
            # Re-run discriminator on the same fake_images (with grad)
            fake_pred, fake_features = self.gan.discriminator(fake_images, return_features=True)

            # Real features for feature matching — no grad needed on real side
            with torch.no_grad():
                _, real_features = self.gan.discriminator(real_images, return_features=True)

            g_loss, g_metrics = self.loss_calculator.compute_generator_loss(
                fake_pred,
                real_images,
                fake_images,
                real_features,
                fake_features
            )

        # FIX 6: Skip backward if loss is still NaN after all guards
        if torch.isnan(g_loss) or torch.isinf(g_loss):
            print(f"\n⚠ NaN G loss at step {self.global_step} — skipping G backward")
            g_metrics['total'] = float('nan')
            return g_metrics

        # Backward + FIX 5: clip gradients
        if self.use_mixed_precision:
            self.scaler.scale(g_loss).backward()
            self.scaler.unscale_(self.optimizer_G)
            torch.nn.utils.clip_grad_norm_(
                self.gan.generator.parameters(), max_norm=GRAD_CLIP_NORM
            )
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.gan.generator.parameters(), max_norm=GRAD_CLIP_NORM
            )
            self.optimizer_G.step()

        return g_metrics

    # ==================== Validation ====================

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.gan.generator.eval()
        self.gan.discriminator.eval()

        val_metrics = defaultdict(list)

        for real_images, _ in self.val_loader:
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)

            z = torch.randn(batch_size, config.LATENT_DIM).to(self.device)
            fake_images = self.gan.generator(z)

            if torch.isnan(fake_images).any():
                continue

            real_pred, real_features = self.gan.discriminator(real_images, return_features=True)
            fake_pred, fake_features = self.gan.discriminator(fake_images, return_features=True)

            _, d_dict = self.loss_calculator.compute_discriminator_loss(real_pred, fake_pred)
            _, g_dict = self.loss_calculator.compute_generator_loss(
                fake_pred, real_images, fake_images, real_features, fake_features
            )

            for key, value in d_dict.items():
                if not np.isnan(value):
                    val_metrics[f'd_{key}'].append(value)
            for key, value in g_dict.items():
                if not np.isnan(value):
                    val_metrics[f'g_{key}'].append(value)

        avg_metrics = {
            key: float(np.mean(values)) if values else float('nan')
            for key, values in val_metrics.items()
        }
        return avg_metrics

    # ==================== Helpers ====================

    def _update_learning_rates(self, val_metrics: Optional[Dict] = None):
        """
        FIX 7: Scheduler.step() called AFTER optimizer.step().
        This is the correct order per PyTorch docs (1.1.0+).
        The warning seen in training output was caused by calling scheduler
        before optimizer in the original code.
        """
        if self.scheduler_G is not None:
            if isinstance(self.scheduler_G, optim.lr_scheduler.ReduceLROnPlateau):
                if val_metrics:
                    self.scheduler_G.step(val_metrics.get('g_total', 0))
            else:
                self.scheduler_G.step()

        if self.scheduler_D is not None:
            if isinstance(self.scheduler_D, optim.lr_scheduler.ReduceLROnPlateau):
                if val_metrics:
                    self.scheduler_D.step(val_metrics.get('d_total', 0))
            else:
                self.scheduler_D.step()

    def _log_batch_metrics(self, d_metrics: Dict, g_metrics: Optional[Dict] = None):
        if not self.writer:
            return
        for key, value in d_metrics.items():
            if not np.isnan(value):
                self.writer.add_scalar(f'Train_Batch/D_{key}', value, self.global_step)
        if g_metrics:
            for key, value in g_metrics.items():
                if not np.isnan(value):
                    self.writer.add_scalar(f'Train_Batch/G_{key}', value, self.global_step)

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ):
        d_train = train_metrics.get('d_total', float('nan'))
        g_train = train_metrics.get('g_total', float('nan'))
        d_val = val_metrics.get('d_total', float('nan'))
        g_val = val_metrics.get('g_total', float('nan'))

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"  Train — D_loss: {d_train:.4f}, G_loss: {g_train:.4f}")
        print(f"  Val   — D_loss: {d_val:.4f}, G_loss: {g_val:.4f}")

        if self.writer:
            for key, value in train_metrics.items():
                if not np.isnan(value):
                    self.writer.add_scalar(f'Train_Epoch/{key}', value, epoch)
            for key, value in val_metrics.items():
                if not np.isnan(value):
                    self.writer.add_scalar(f'Val_Epoch/{key}', value, epoch)
            self.writer.add_scalar('LR/generator', self.optimizer_G.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('LR/discriminator', self.optimizer_D.param_groups[0]['lr'], epoch)

        for key, value in train_metrics.items():
            self.train_metrics[key].append(value)
        for key, value in val_metrics.items():
            self.val_metrics[key].append(value)

    @torch.no_grad()
    def _generate_visualizations(self, epoch: int):
        self.gan.generator.eval()

        num_samples = 16
        z = torch.randn(num_samples, config.LATENT_DIM).to(self.device)
        fake_images = self.gan.generator(z)

        save_path = config.RESULTS_DIR / 'training_progress' / f'epoch_{epoch+1}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot_vein_batch(
            fake_images,
            title=f"Generated Samples - Epoch {epoch+1}",
            save_path=save_path
        )

        if self.writer:
            from torchvision.utils import make_grid
            grid = make_grid((fake_images + 1) / 2, nrow=4, padding=2, normalize=False)
            self.writer.add_image('Generated_Samples', grid, epoch)

        real_images, _ = next(iter(self.val_loader))
        real_images = real_images[:num_samples].to(self.device)
        comparison_path = (
            config.RESULTS_DIR / 'training_progress' / f'comparison_epoch_{epoch+1}.png'
        )
        compare_real_fake(real_images, fake_images, save_path=comparison_path)

    def _save_checkpoint(self, filename: str, is_best: bool = False):
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
            'loss_weights': self.loss_calculator.weights,
            'config': {
                'latent_dim': config.LATENT_DIM,
                'image_size': config.IMAGE_SIZE,
                'batch_size': config.BATCH_SIZE,
            }
        }

        if self.scheduler_G is not None:
            checkpoint['scheduler_g_state_dict'] = self.scheduler_G.state_dict()
        if self.scheduler_D is not None:
            checkpoint['scheduler_d_state_dict'] = self.scheduler_D.state_dict()

        # FIX 1: Save single scaler state
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        history_path = config.CHECKPOINT_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train_metrics': {
                    k: [float(v) for v in vals]
                    for k, vals in self.train_metrics.items()
                },
                'val_metrics': {
                    k: [float(v) for v in vals]
                    for k, vals in self.val_metrics.items()
                }
            }, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_d_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']

        # Restore loss weights if saved
        if 'loss_weights' in checkpoint:
            self.loss_calculator.update_weights(checkpoint['loss_weights'])

        if 'train_metrics' in checkpoint:
            self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
        if 'val_metrics' in checkpoint:
            self.val_metrics = defaultdict(list, checkpoint['val_metrics'])

        if self.scheduler_G and 'scheduler_g_state_dict' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_g_state_dict'])
        if self.scheduler_D and 'scheduler_d_state_dict' in checkpoint:
            self.scheduler_D.load_state_dict(checkpoint['scheduler_d_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best metric: {self.best_metric:.4f}")


def resume_training(checkpoint_path: str, gan, train_loader, val_loader):
    trainer = VeinGANTrainer(gan, train_loader, val_loader)
    trainer.load_checkpoint(checkpoint_path)
    trainer.train()


if __name__ == "__main__":
    print("Testing VeinGANTrainer...")

    from models.gan import VeinGAN
    from data.data_loader import create_data_loaders

    print("Creating data loaders...")
    db_paths = [config.DB1_PATH, config.DB2_PATH]
    train_loader, val_loader, test_loader = create_data_loaders(db_paths)

    print("Creating GAN...")
    gan = VeinGAN()

    print("Creating trainer...")
    trainer = VeinGANTrainer(gan, train_loader, val_loader)

    print("\n✓ Trainer initialized successfully!")
    print(f"  Ready to train for {config.NUM_EPOCHS} epochs")