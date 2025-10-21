# src/train.py

# --- Gürültülü TF loglarını sustur (bazı bağımlılıklar TF'ü dolaylı yüklüyor olabilir) ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import torchvision.transforms as transforms
import sys
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, Resized

# --- TQDM KÜTÜPHANESİNİ İÇERİ AKTARMA ---
from tqdm import tqdm

# Proje ana dizinini import yollarına ekleyelim
PROJE_ANA_DIZINI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_ANA_DIZINI not in sys.path:
    sys.path.append(PROJE_ANA_DIZINI)

from src.data_utils import HeadCTDatasetMONAI as HeadCTDataset  # :contentReference[oaicite:0]{index=0}
from src.data_utils import collate_fn_skip_corrupted            # :contentReference[oaicite:1]{index=1}
from src.model import get_model                                 # :contentReference[oaicite:2]{index=2}

# --- 1. AYARLAR VE HİPERPARAMETRELER ---
DATA_PATH = 'Data/processed/master_image_list.csv'
MODEL_SAVE_PATH = 'models/'
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

def main():
    # --- 2. VERİ HAZIRLAMA ---
    print("Veri hazırlanıyor...")
    df_master = pd.read_csv(DATA_PATH)

    # ✅ Path düzeltmesi: src'ten yukarı çıkıp gerçek Data klasörünü baz al
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data")  # .../Prototype/Data

    def fix_path(x: str) -> str:
        # Sadece ../Data veya ..\Data ile başlayan pathleri normalize et
        if x.startswith("../Data/") or x.startswith("..\\Data\\"):
            rel_path = x.replace("../Data/", "").replace("..\\Data\\", "")
            fixed = os.path.normpath(os.path.join(DATA_DIR, rel_path))
            return fixed
        # Tam yol ya da farklı bir göreli yol ise olduğu gibi normalize et
        return os.path.normpath(x)

    df_master["dicom_path"] = df_master["dicom_path"].apply(fix_path)

    # ✅ Label tipi garanti (CrossEntropyLoss için long/int sınıf indexi gerekli)
    df_master["is_hemorrhage"] = df_master["is_hemorrhage"].astype(int)

    # 🔍 Örnek path kontrolü
    print("Örnek DICOM yolları:", df_master["dicom_path"].head(3).tolist())

    train_df, val_df = train_test_split(
        df_master,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=df_master['is_hemorrhage']
    )
    
    print(f"Eğitim verisi boyutu: {len(train_df)}")
    print(f"Validasyon verisi boyutu: {len(val_df)}")

    # --- MONAI transforms ---
    data_transforms = Compose([
        LoadImaged(keys='image', reader="PydicomReader", image_only=True),
        EnsureChannelFirstd(keys='image'),
        Resized(keys='image', spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        ScaleIntensityRanged(keys='image', a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True),
        ToTensord(keys='image'),
    ])

    train_dataset = HeadCTDataset(train_df.reset_index(drop=True), transform=data_transforms)  # __getitem__ label->long :contentReference[oaicite:3]{index=3}
    val_dataset   = HeadCTDataset(val_df.reset_index(drop=True),   transform=data_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_skip_corrupted,  # Bozuk örnekleri batch dışına atar :contentReference[oaicite:4]{index=4}
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_skip_corrupted,
        num_workers=2,
        pin_memory=True
    )

    # --- 3. MODEL VE EĞİTİM ARAÇLARI ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılacak cihaz: {device}")

    model = get_model(pretrained=True, num_classes=2).to(device)  # ResNet18, 1-kanal giriş + 2 sınıf :contentReference[oaicite:5]{index=5}
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 4. EĞİTİM VE VALİDASYON DÖNGÜSÜ ---
    best_val_accuracy = 0.0
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        # --- Eğitim ---
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Eğitim]"):
            # Boş batch korunması
            if images.nelement() == 0:
                continue

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        # --- Validasyon ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validasyon]"):
                if images.nelement() == 0:
                    continue

                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        # Epoch istatistikleri
        avg_train_loss = train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else float('nan')
        train_accuracy = train_correct / len(train_loader.dataset) if len(train_loader.dataset) > 0 else float('nan')
        avg_val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('nan')
        val_accuracy = val_correct / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('nan')

        # Debug özetleri
        print(f"DEBUG: train_correct={train_correct}, train_size={len(train_loader.dataset)}")
        print(f"DEBUG: val_correct={val_correct}, val_size={len(val_loader.dataset)}")

        print(f"Eğitim Kaybı: {avg_train_loss:.4f} | Eğitim Doğruluğu: {train_accuracy:.4f}")
        print(f"Validasyon Kaybı: {avg_val_loss:.4f} | Validasyon Doğruluğu: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))
            print(f"✨ Yeni en iyi model kaydedildi! Validasyon Doğruluğu: {best_val_accuracy:.4f}")

    print("\n✅ Eğitim tamamlandı!")

if __name__ == '__main__':
    main()
