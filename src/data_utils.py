# src/data_utils.py

import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import os # os modülünü import edelim

class HeadCTDataset(Dataset):
    """Kafa BT DICOM görüntüleri için özel PyTorch Dataset sınıfı."""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # CSV'den orijinal yolu al
        img_path_relative = self.df.loc[idx, 'dicom_path']
        label = self.df.loc[idx, 'is_hemorrhage']
        
        # --- YENİ EKLENEN SATIR ---
        # Yolun başındaki '../' kısmını kaldırarak yolu düzelt.
        # Bu, script'in proje ana dizininden çalıştırılmasıyla uyumlu hale getirir.
        img_path = img_path_relative.lstrip('../')
        # -------------------------

        try:
            # Not: Proje ana dizinini de ekleyerek tam yolu oluşturabiliriz, ama lstrip genellikle yeterlidir.
            # PROJE_ANA_DIZINI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # full_path = os.path.join(PROJE_ANA_DIZINI, img_path)
            
            dcm_file = pydicom.dcmread(img_path) # Düzeltilmiş yolu kullan
            image = dcm_file.pixel_array
            
            if image.ndim > 2:
                image = image[:, :, 0]
            
        except Exception as e:
            return None, None

        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
        image = (image * 255).astype(np.uint8)
        
        if self.transform:
            image = self.transform(image)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image, label_tensor

def collate_fn_skip_corrupted(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)