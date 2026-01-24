"""
EfficientNetV2 Model for Pill Image Classification
Uses transfer learning from ImageNet pretrained weights
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class EfficientNetPillClassifier(nn.Module):
    """
    EfficientNetV2-based classifier for pill image identification.

    Args:
        num_classes: Number of pill classes (4902 for ePillID dataset)
        model_name: EfficientNetV2 variant ('efficientnetv2_s', 'm', 'l')
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout probability for regularization
        freeze_backbone: Whether to freeze backbone weights initially
    """

    def __init__(
        self,
        num_classes: int = 960,  # Actual ePillID dataset size (will be overridden per fold)
        model_name: str = "efficientnetv2_s",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained EfficientNetV2 from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classifier layers with Xavier/Kaiming initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_bn(self):
        """Freeze BatchNorm layers in backbone for more stable training."""
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def get_trainable_params(self) -> list:
        """Get list of trainable parameter names."""
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def get_param_groups(self, lr: float = 1e-3, backbone_lr_multiplier: float = 0.1):
        """
        Get parameter groups for differential learning rates.

        Args:
            lr: Base learning rate for classifier
            backbone_lr_multiplier: Multiplier for backbone LR (usually < 1)

        Returns:
            List of parameter group dicts
        """
        backbone_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)

        param_groups = [
            {'params': classifier_params, 'lr': lr, 'name': 'classifier'},
            {
                'params': backbone_params,
                'lr': lr * backbone_lr_multiplier,
                'name': 'backbone'
            }
        ]

        # Remove empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        return param_groups


class EfficientNetPillClassifierWithArcFace(nn.Module):
    """
    EfficientNetV2 with ArcFace loss for better embedding learning.
    ArcFace (Additive Angular Margin Loss) improves feature discrimination.
    """

    def __init__(
        self,
        num_classes: int = 960,  # Actual ePillID dataset size (will be overridden per fold)
        model_name: str = "efficientnetv2_s",
        pretrained: bool = True,
        feature_dim: int = 512,
        scale: float = 30.0,
        margin: float = 0.5
    ):
        super().__init__()

        self.num_classes = num_classes

        # Backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        backbone_dim = self.backbone.num_features

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # ArcFace will be applied during training
        self.scale = scale
        self.margin = margin

        # Classification head (for inference)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass.

        Args:
            x: Input images
            return_features: If True, return features along with logits

        Returns:
            Logits (and optionally features)
        """
        # Extract features
        features = self.backbone(x)
        features = self.feature_proj(features)

        # Normalize features for ArcFace
        features_norm = nn.functional.normalize(features, p=2, dim=1)

        # Get logits
        weight = self.classifier.weight
        weight_norm = nn.functional.normalize(weight, p=2, dim=1)
        logits = nn.functional.linear(features_norm, weight_norm) * self.scale

        if return_features:
            return logits, features
        return logits


def create_model(
    num_classes: int = 4902,
    model_size: str = "s",
    pretrained: bool = True,
    use_arcface: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a pill classifier model.

    Args:
        num_classes: Number of output classes
        model_size: Model size/name - 's', 'm', 'l', or direct model name like 'tf_efficientnetv2_s', 'efficientnet_b0'
        pretrained: Use ImageNet pretrained weights
        use_arcface: Use ArcFace variant
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model
    """
    # Map short names to actual model names
    model_name_map = {
        's': 'tf_efficientnetv2_s',
        'm': 'tf_efficientnetv2_m',
        'l': 'tf_efficientnetv2_l',
        'b0': 'efficientnet_b0',
        'b1': 'efficientnet_b1',
    }

    # Get actual model name
    if model_size in model_name_map:
        model_name = model_name_map[model_size]
    else:
        # Assume it's already a valid model name
        model_name = model_size

    if use_arcface:
        model = EfficientNetPillClassifierWithArcFace(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )
    else:
        model = EfficientNetPillClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )

    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the model.

    Returns:
        Dict with total_params, trainable_params, model_size_mb
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Approximate model size in MB
    model_size_mb = total_params * 4 / (1024 * 1024)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'model_name': model.model_name if hasattr(model, 'model_name') else 'Unknown'
    }


if __name__ == "__main__":
    # Test model creation
    print("Creating EfficientNetV2-S model for 4902 classes...")

    model = create_model(
        num_classes=4902,
        model_size='s',
        pretrained=False
    )

    info = get_model_info(model)
    print(f"\nModel Info:")
    for k, v in info.items():
        print(f"  {k}: {v:,}" if 'mb' not in k else f"  {k}: {v:.2f}")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
