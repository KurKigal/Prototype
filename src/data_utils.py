# src/data_utils.py

import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import os

# MONAI'den gerekli modülleri import et
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Resized

class HeadCTDatasetMONAI(Dataset):
    """Kafa BT DICOM görüntüleri için MONAI dönüşümlerini kullanan Dataset sınıfı."""
    
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): 'dicom_path' ve 'is_hemorrhage' sütunlarını içeren dataframe.
            transform (callable, optional): Görüntüye uygulanacak MONAI dönüşüm zinciri.
        """
        # Veriyi MONAI'nin beklediği sözlük formatına hazırlayalım
        # [{'image': 'path/to/img1.dcm', 'label': 0}, {'image': 'path/to/img2.dcm', 'label': 1}, ...]
        self.file_list = dataframe.rename(columns={'dicom_path': 'image', 'is_hemorrhage': 'label'}).to_dict('records')
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # idx'deki sözlüğü al
        data_item = self.file_list[idx]
        
        # --- ÖNEMLİ DEĞİŞİKLİK ---
        # DICOM okumayı ve temel işlemleri MONAI transformlarına bırakıyoruz.
        # Bu, daha temiz ve MONAI standardına uygun bir yoldur.
        
        # 'image' anahtarını (dosya yolu) içeren bir sözlük oluştur
        # Dönüşüm zinciri bu sözlüğü işleyecek
        data_dict = {'image': data_item['image'], 'label': data_item['label']}

        # Tanımlanmışsa dönüşüm zincirini uygula
        if self.transform:
            try:
                data_dict = self.transform(data_dict)
            except Exception as e:
                # print(f"HATA: {data_item['image']} işlenirken hata oluştu: {e}")
                # Hata durumunda boş veri döndürerek DataLoader'ın devam etmesini sağla
                # Not: Hata ayıklama sırasında bu print'i açabilirsiniz.
                return None # collate_fn None'ları atlayacak

        # Dönüştürülmüş tensörleri ve etiketi döndür
        # Eğer etiket de dönüştürüldüyse (örn: one-hot encoding), ona göre ayarlama gerekebilir.
        # Şimdilik etiketin skaler olduğunu varsayıyoruz.
        return data_dict['image'], torch.tensor(data_dict['label'], dtype=torch.long)


# Hatalı (None dönen) verileri ayıklamak için collate_fn
def collate_fn_skip_corrupted(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([]) # Boş tensörler döndür
    # PyTorch'un varsayılan collate fonksiyonunu kullanarak batch'i birleştir
    return torch.utils.data.dataloader.default_collate(batch)