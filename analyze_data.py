import os
from PIL import Image
from collections import Counter

def analyze_data(root='training_data'):
    total_pngs = 0
    size_counter = Counter()
    sample_paths = []
    category_counter = Counter()
    
    for p, _, files in os.walk(root):
        for f in files:
            if f.endswith('.png'):
                total_pngs += 1
                path = os.path.join(p, f)
                if len(sample_paths) < 10:
                    sample_paths.append(path)
                
                # 统计图片尺寸
                try:
                    with Image.open(path) as img:
                        size_counter[img.size] += 1
                except:
                    continue
                
                # 尝试提取类别（比如 block, item, entity）
                parts = path.split(os.sep)
                for cat in ['block', 'item', 'entity', 'particle', 'effect']:
                    if cat in parts:
                        category_counter[cat] += 1
                        break

    print("--- Dataset Analysis Report ---")
    print(f"Total PNG files found: {total_pngs}")
    print("\nSize Distribution (Top 5):")
    for size, count in size_counter.most_common(5):
        print(f"  {size}: {count} images")
    
    print("\nCategory Distribution:")
    for cat, count in category_counter.most_common():
        print(f"  {cat}: {count} images")
        
    print("\nPath structure samples:")
    for path in sample_paths:
        print(f"  {path}")

if __name__ == "__main__":
    analyze_data()
