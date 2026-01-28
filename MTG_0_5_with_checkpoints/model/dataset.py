import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MCTextureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.texts = []
        
        print("🚀 [Industrial Scanner] Starting deep scan of all mods...")
        
        # 允许的类别关键字
        allowed_cats = {'block', 'item', 'particle', 'effect', 'mob_effect'}
        
        for root, dirs, files in os.walk(root_dir):
            # 获取路径层级
            relative_path = os.path.relpath(root, root_dir)
            parts = relative_path.split(os.sep)
            
            # 基础信息提取
            # 假设第一层是材质包名，之后某一层是类别
            pack_name = parts[0] if parts[0] != '.' else 'vanilla'
            
            # 寻找类别标签
            category = "unknown"
            for p in parts:
                if p.lower() in allowed_cats:
                    category = p.lower()
                    break
            
            for filename in files:
                if filename.endswith('.png') and not filename.endswith('.mcmeta'):
                    full_path = os.path.join(root, filename)
                    try:
                        # 极其重要的预检查：只取 16x16，且必须是正方形
                        with Image.open(full_path) as img:
                            if img.size != (16, 16):
                                continue
                        
                        self.image_paths.append(full_path)
                        
                        # 标签处理逻辑：
                        # 1. 把文件名中的 _ 换成空格
                        # 2. 拼接: 材质包名 + 类别 + 文件名
                        clean_name = filename.replace('.png', '').replace('_', ' ').replace('-', ' ')
                        label = f"{pack_name} {category} {clean_name}"
                        self.texts.append(label.lower())
                        
                    except Exception:
                        continue
        
        print(f"✅ Scan Complete: Found {len(self.image_paths)} valid 16x16 textures.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        text = self.texts[idx]
        
        # 加载并归一化
        img = Image.open(path).convert('RGBA')
        
        # 数据增强 (可选，但推荐)
        # 对像素艺术而言，左右翻转是唯一不破坏语义的增强
        if torch.rand(1) < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        ])
        return transform(img), text