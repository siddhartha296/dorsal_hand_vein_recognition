"""
Vein Authentication Classifier
Used for authentication tasks and XAI analysis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import config


class VeinAuthenticationClassifier(nn.Module):
    """
    CNN Classifier for vein authentication
    Can be used standalone or as part of the GAN evaluation
    """
    
    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int] = None,
        input_channels: int = 1
    ):
        """
        Args:
            num_classes: Number of unique persons to classify
            image_size: Input image size (H, W)
            input_channels: Number of input channels (1 for grayscale)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size or config.IMAGE_SIZE
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Adaptive pooling for flexible input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            Class logits (B, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representation before classification
        
        Args:
            x: Input image
        
        Returns:
            Feature vector
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Pass through first FC layers but not final classifier
        x = self.fc[1](x)  # Linear
        x = self.fc[2](x)  # ReLU
        x = self.fc[4](x)  # Linear
        x = self.fc[5](x)  # ReLU
        
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities
        
        Args:
            x: Input image
        
        Returns:
            (predicted_classes, probabilities)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


class SiameseVeinNetwork(nn.Module):
    """
    Siamese Network for vein verification
    Compares two vein images and outputs similarity score
    """
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding for one image"""
        x = self.feature_extractor(x)
        x = self.embedding(x)
        return x
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for image pair
        
        Args:
            x1, x2: Image pairs
        
        Returns:
            (embedding1, embedding2)
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2
    
    def compute_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between image pairs
        
        Returns:
            Similarity scores
        """
        emb1, emb2 = self.forward(x1, x2)
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese network training
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emb1, emb2: Embeddings from Siamese network
            label: 1 for same person, 0 for different persons
        
        Returns:
            Contrastive loss
        """
        euclidean_distance = F.pairwise_distance(emb1, emb2)
        
        loss_positive = label * torch.pow(euclidean_distance, 2)
        loss_negative = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        loss = torch.mean(loss_positive + loss_negative)
        return loss


def test_classifier():
    """Test the classifier"""
    print("Testing VeinAuthenticationClassifier...")
    
    # Create classifier
    num_classes = 100
    classifier = VeinAuthenticationClassifier(num_classes=num_classes)
    
    # Test forward pass
    batch_size = 4
    dummy_images = torch.randn(batch_size, 1, *config.IMAGE_SIZE)
    
    with torch.no_grad():
        # Standard forward
        logits = classifier(dummy_images)
        print(f"Input shape: {dummy_images.shape}")
        print(f"Output logits shape: {logits.shape}")
        
        # Extract features
        features = classifier.extract_features(dummy_images)
        print(f"Feature vector shape: {features.shape}")
        
        # Make predictions
        preds, probs = classifier.predict(dummy_images)
        print(f"Predictions shape: {preds.shape}")
        print(f"Probabilities shape: {probs.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in classifier.parameters())
    print(f"Total parameters: {num_params:,}")
    
    return classifier


if __name__ == "__main__":
    test_classifier()
