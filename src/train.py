# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import torchvision.transforms as transforms
import os
import sys

# --- TQDM KÜTÜPHANESİNİ İÇERİ AKTARMA ---
from tqdm import tqdm

# Proje ana dizinini import yollarına ekleyelim
PROJE_ANA_DIZINI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_ANA_DIZINI not in sys.path:
    sys.path.append(PROJE_ANA_DIZINI)

from src.data_utils import HeadCTDataset, collate_fn_skip_corrupted
from src.model import get_model

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
    train_df, val_df = train_test_split(df_master, test_size=VALIDATION_SPLIT, random_state=42, stratify=df_master['is_hemorrhage'])
    
    print(f"Eğitim verisi boyutu: {len(train_df)}")
    print(f"Validasyon verisi boyutu: {len(val_df)}")

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = HeadCTDataset(train_df.reset_index(drop=True), transform=data_transforms)
    val_dataset = HeadCTDataset(val_df.reset_index(drop=True), transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_skip_corrupted, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_skip_corrupted, num_workers=2, pin_memory=True)

    # --- 3. MODEL VE EĞİTİM ARAÇLARINI KURMA ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılacak cihaz: {device}")

    model = get_model(pretrained=True, num_classes=2).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 4. EĞİTİM VE VALİDASYON DÖNGÜSÜ ---
    best_val_accuracy = 0.0
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # --- EĞİTİM DÖNGÜSÜNÜ TQDM İLE SARMA ---
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Eğitim]"):
            images, labels = images.to(device), labels.to(device)
            if images.nelement() == 0: continue # Boş batch'leri atla
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        # --- VALİDASYON DÖNGÜSÜNÜ TQDM İLE SARMA ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validasyon]"):
                images, labels = images.to(device), labels.to(device)
                if images.nelement() == 0: continue # Boş batch'leri atla

                outputs = model(images)
                loss = loss_function(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        # Epoch sonu istatistikleri
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct.double() / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct.double() / len(val_loader.dataset)
        
        print(f"Eğitim Kaybı: {avg_train_loss:.4f} | Eğitim Doğruluğu: {train_accuracy:.4f}")
        print(f"Validasyon Kaybı: {avg_val_loss:.4f} | Validasyon Doğruluğu: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))
            print(f"✨ Yeni en iyi model kaydedildi! Validasyon Doğruluğu: {best_val_accuracy:.4f}")

    print("\n✅ Eğitim tamamlandı!")

if __name__ == '__main__':
    main()