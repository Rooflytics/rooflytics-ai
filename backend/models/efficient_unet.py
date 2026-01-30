import segmentation_models_pytorch as smp
import torch.nn as nn


def get_efficientnet_unet(num_classes: int = 1):
    """
    EfficientNet-B0 U-Net with ImageNet pretrained encoder.
    """
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,  # logits
    )
    return model
