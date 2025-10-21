import torch
import torch.nn as nn
from torchvision import models

def get_model(pretrained=True, num_classes=2):
    """
    Önceden eğitilmiş bir ResNet18 modeli yükler ve projemize uygun hale getirir.
    
    Args:
        pretrained (bool): ImageNet üzerinde önceden eğitilmiş ağırlıkları kullanıp kullanmayacağımız.
        num_classes (int): Çıktı sınıflarının sayısı.
    """

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    original_weights = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    with torch.no_grad():
        model.conv1.weight[:, 0] = original_weights.mean(1)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model