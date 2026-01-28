import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import MCTextureDataset
from .networks import ConditionalUNet
from .config import DEVICE, XPU_AVAILABLE
import os

def train():
    dataset = MCTextureDataset(root_dir='training_data')
    if len(dataset) == 0: return

    # --- 构建词表 (Tokenizer) ---
    all_words = set()
    for text in dataset.texts:
        for word in text.split():
            all_words.add(word)
    
    vocab = {word: i + 1 for i, word in enumerate(sorted(list(all_words)))} # 0 留给 Padding
    vocab_size = len(vocab)
    print(f"Vocab Size: {vocab_size} unique words.")

    def tokenize(texts, max_len=10):
        res = torch.zeros((len(texts), max_len), dtype=torch.long)
        for i, text in enumerate(texts):
            words = text.split()
            for j, word in enumerate(words[:max_len]):
                res[i, j] = vocab.get(word, 0)
        return res

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = ConditionalUNet(vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # 热重启调度器：每 100 轮重置一次学习率，帮助跳出局部最优解
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)
    
    n_steps = 1000
    beta = torch.linspace(1e-4, 0.02, n_steps).to(DEVICE)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    
    print(f"Training on {DEVICE}...")
    scaler = torch.amp.GradScaler('xpu') if XPU_AVAILABLE else None

    epoch = 0
    target_loss = 0.005
    
    while True:
        epoch_loss = 0
        model.train()
        for images, texts in dataloader:
            images = images.to(DEVICE)
            # Tokenize 现在的三段式文本
            token_ids = tokenize(texts).to(DEVICE)
            
            # Classifier-Free Guidance: 15% 概率完全丢弃文本信息 (设为 0)
            if torch.rand(1) < 0.15:
                token_ids = torch.zeros_like(token_ids)
            
            t = torch.randint(0, n_steps, (images.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(images)
            a_cp = alpha_cumprod[t][:, None, None, None]
            noisy_images = torch.sqrt(a_cp) * images + torch.sqrt(1 - a_cp) * noise
            
            optimizer.zero_grad()
            if XPU_AVAILABLE:
                with torch.amp.autocast(device_type='xpu'):
                    predicted_noise = model(noisy_images, t, token_ids)
                    loss = nn.MSELoss()(predicted_noise, noise)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predicted_noise = model(noisy_images, t, token_ids)
                loss = nn.MSELoss()(predicted_noise, noise)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6e}")
        
        # 每 50 轮保存一次
        if epoch % 50 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'epoch': epoch,
                'loss': avg_loss
            }, f"model/checkpoint_v2_epoch_{epoch}.pth")

        if avg_loss < target_loss:
            print(f"Target loss {target_loss} reached at epoch {epoch}!")
            torch.save(model.state_dict(), "model/final_model.pth")
            break
            
        epoch += 1

if __name__ == "__main__":
    train()