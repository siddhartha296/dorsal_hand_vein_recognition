"""
Training loop for XAI-Enhanced GAN
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .losses import GANLosses
from config import config

class VeinGANTrainer:
    def __init__(self, gan, train_loader, val_loader):
        self.gan = gan
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.DEVICE
        
        self.optimizer_G = optim.Adam(
            self.gan.generator.parameters(), 
            lr=config.LEARNING_RATE_G, 
            betas=(config.BETA1, config.BETA2)
        )
        self.optimizer_D = optim.Adam(
            self.gan.discriminator.parameters(), 
            lr=config.LEARNING_RATE_D, 
            betas=(config.BETA1, config.BETA2)
        )
        
        self.losses = GANLosses(device=self.device)
        self.writer = SummaryWriter(log_dir=config.LOGS_DIR)

    def train_epoch(self, epoch):
        self.gan.generator.train()
        self.gan.discriminator.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for i, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # --- Train Discriminator ---
            self.optimizer_D.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, config.LATENT_DIM, 1, 1).to(self.device)
            fake_images = self.gan.generator(z)
            
            # D Loss
            d_real = self.gan.discriminator(real_images)
            d_fake = self.gan.discriminator(fake_images.detach())
            
            loss_d_real = self.losses.get_adversarial_loss(d_real, True)
            loss_d_fake = self.losses.get_adversarial_loss(d_fake, False)
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            loss_d.backward()
            self.optimizer_D.step()
            
            # --- Train Generator ---
            if i % config.N_CRITIC == 0:
                self.optimizer_G.zero_grad()
                
                d_fake_for_g = self.gan.discriminator(fake_images)
                
                # Combined G Loss
                loss_g_adv = self.losses.get_adversarial_loss(d_fake_for_g, True)
                loss_g_perceptual = self.losses.get_perceptual_loss(real_images, fake_images)
                
                loss_g = (config.ADVERSARIAL_LOSS_WEIGHT * loss_g_adv + 
                          config.PERCEPTUAL_LOSS_WEIGHT * loss_g_perceptual)
                
                loss_g.backward()
                self.optimizer_G.step()
                
                pbar.set_postfix({"D_loss": loss_d.item(), "G_loss": loss_g.item()})

    def train(self):
        for epoch in range(config.NUM_EPOCHS):
            self.train_epoch(epoch)
            
            if epoch % config.SAVE_INTERVAL == 0:
                ckpt_path = config.CHECKPOINT_DIR / f"gan_epoch_{epoch}.pth"
                self.gan.save_checkpoint(ckpt_path, epoch, self.optimizer_G, self.optimizer_D)