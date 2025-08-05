import os
import shutil
import random

def list_category_dirs(raw_dir):
    return [
        os.path.join(raw_dir, name)
        for name in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, name))
    ]

def count_images(raw_dir):
    counts = {}
    for category_path in list_category_dirs(raw_dir):
        name = os.path.basename(category_path)
        files = [
            f for f in os.listdir(category_path)
            if os.path.isfile(os.path.join(category_path, f))
        ]
        counts[name] = len(files)
    return counts

def create_split_dirs(base_dir, splits, categories):
    for split in splits:
        for category in categories:
            path = os.path.join(base_dir, split, category)
            os.makedirs(path, exist_ok=True)

def split_and_move(raw_dir, base_dir, train_ratio, val_ratio):
    splits = ['train', 'val', 'test']
    categories = [os.path.basename(p) for p in list_category_dirs(raw_dir)]
    create_split_dirs(base_dir, splits, categories)
    for category in categories:
        source = os.path.join(raw_dir, category)
        images = [
            f for f in os.listdir(source)
            if os.path.isfile(os.path.join(source, f))
        ]
        random.shuffle(images)
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        allocations = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        for split, files in allocations.items():
            for filename in files:
                src = os.path.join(source, filename)
                dst = os.path.join(base_dir, split, category, filename)
                shutil.move(src, dst)
        if not os.listdir(source):
            os.rmdir(source)
    if not os.listdir(raw_dir):
        os.rmdir(raw_dir)

if __name__ == '__main__':
    raw_directory = 'dataset/raw'
    base_directory = 'dataset'
    ratios = (0.7, 0.15)
    counts = count_images(raw_directory)
    for category, count in counts.items():
        print(f"{category}: {count} images")
    split_and_move(raw_directory, base_directory, ratios[0], ratios[1])
