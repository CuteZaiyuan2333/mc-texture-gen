import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DEVICE

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        # FiLM 强耦合：生成缩放 (gamma) 和 偏移 (beta)
        self.film_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        # FiLM 注入
        film_params = self.film_mlp(t)[:, :, None, None]
        gamma, beta = film_params.chunk(2, dim=1)
        h = h * (1 + gamma) + beta # 强行改变特征图的分布
        
        h = self.conv2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)
        x_norm = self.ln(x_flat)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        return x + attn_out.transpose(1, 2).view(b, c, h, w)

class ConditionalUNet(nn.Module):
    def __init__(self, vocab_size, time_emb_dim=512): # 扩大到 512 增强记忆力
        super().__init__()
        self.label_emb = nn.Embedding(vocab_size + 1, time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.inc = nn.Conv2d(4, 128, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(128, 256, time_emb_dim)
        self.down2 = ResidualBlock(256, 512, time_emb_dim)
        
        self.mid1 = ResidualBlock(512, 512, time_emb_dim)
        self.attn = AttentionBlock(512)
        self.mid2 = ResidualBlock(512, 512, time_emb_dim)
        
        self.up1 = ResidualBlock(512 + 256, 256, time_emb_dim)
        self.up2 = ResidualBlock(256 + 128, 128, time_emb_dim)
        self.outc = nn.Conv2d(128, 4, kernel_size=1)

    def forward(self, x, t, token_ids):
        # 优化：采用求和而非平均，保留更多特征强度
        mask = (token_ids != 0).float()
        l_embs = self.label_emb(token_ids)
        l_emb = (l_embs * mask.unsqueeze(-1)).sum(dim=1) 
        
        t_emb = self.pos_encoding(t, 512)
        emb = self.time_mlp(t_emb + l_emb)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, emb)
        x3 = self.down2(F.avg_pool2d(x2, 2), emb)
        
        x3 = self.mid1(x3, emb)
        x3 = self.attn(x3)
        x3 = self.mid2(x3, emb)
        
        x = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x, emb)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x, emb)
        return self.outc(x)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=DEVICE).float() / channels))
        pos_enc_a = torch.sin(t[:, None] * inv_freq[None, :])
        pos_enc_b = torch.cos(t[:, None] * inv_freq[None, :])
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)