import torch
import os
import numpy as np
from PIL import Image
from .config import DEVICE, XPU_AVAILABLE
from .networks import ConditionalUNet

def sample(prompt, model_path, output_name="generated_texture.png", cfg_scale=4.0):
    # 1. 加载 Checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    vocab = checkpoint['vocab']
    
    # 2. 初始化模型
    model = ConditionalUNet(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 处理 Prompt (Tokenization)
    def tokenize(text, vocab, max_len=10):
        res = torch.zeros((1, max_len), dtype=torch.long)
        words = text.lower().strip().split()
        for j, word in enumerate(words[:max_len]):
            res[0, j] = vocab.get(word, 0)
        return res

    token_ids = tokenize(prompt, vocab).to(DEVICE)
    uncond_ids = torch.zeros_like(token_ids).to(DEVICE) # 全 0 代表无条件
    
    n_steps = 1000
    beta = torch.linspace(1e-4, 0.02, n_steps).to(DEVICE)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    print(f"Generating '{prompt}' with V2 architecture...")
    
    with torch.no_grad():
        x = torch.randn((1, 4, 16, 16), device=DEVICE)
        for i in reversed(range(n_steps)):
            t = torch.tensor([i], device=DEVICE).long()
            
            # CFG Sampling
            combined_x = torch.cat([x, x], dim=0)
            combined_t = torch.cat([t, t], dim=0)
            combined_l = torch.cat([token_ids, uncond_ids], dim=0)
            
            noise_pred_all = model(combined_x, combined_t, combined_l)
            eps_cond, eps_uncond = noise_pred_all.chunk(2)
            
            predicted_noise = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            a = alpha[i]
            a_cp = alpha_cumprod[i]
            noise = torch.randn_like(x) if i > 0 else 0
            x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - a_cp)) * predicted_noise) + torch.sqrt(beta[i]) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        x = x.cpu().squeeze(0).permute(1, 2, 0).numpy()
        Image.fromarray((x * 255).astype(np.uint8), 'RGBA').resize((256, 256), Image.NEAREST).save(output_name)
        print(f"Success: {output_name}")

if __name__ == "__main__":
    import sys
    # 示例: python -m model.generate "mc-vanilla block stone" model/checkpoint_v2_epoch_50.pth
    p = sys.argv[1] if len(sys.argv) > 1 else "mc-vanilla block acacia door bottom"
    m = sys.argv[2] if len(sys.argv) > 2 else "model/checkpoint_v2_epoch_50.pth"
    sample(p, m)