import os
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torchvision import transforms, models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.paths[idx]

def get_feature_extractor(device):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.eval()
    return m.to(device)

def extract_features(paths, device, batch_size=32, size=224):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = ImagePathDataset(paths, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    ext = get_feature_extractor(device)
    feats, pths = [], []
    with torch.no_grad():
        for imgs, ps in dl:
            f = ext(imgs.to(device)).cpu().numpy()
            feats.append(f); pths.extend(ps)
    return np.vstack(feats), pths

def pick_random_per_folder(root):
    picks = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            imgs = [os.path.join(p, f) for f in os.listdir(p)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if imgs:
                picks[d] = random.choice(imgs)
    return picks

def make_recommendations(root='test', out_dir='recs', topk=3, tile=128):
    os.makedirs(out_dir, exist_ok=True)
    picks = pick_random_per_folder(root)
    all_paths = []
    for r,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                all_paths.append(os.path.join(r,f))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feats, paths = extract_features(all_paths, device)
    sim = cosine_similarity(feats, feats)
    idx = {p:i for i,p in enumerate(paths)}
    font = ImageFont.load_default()
    for cat, qpath in picks.items():
        qi = idx[qpath]
        scores = sim[qi]
        order = np.argsort(scores)[::-1]
        recs = [paths[i] for i in order if paths[i]!=qpath][:topk]
        imgs = [qpath]+recs
        w = tile * len(imgs)
        h = tile + 20
        canvas = Image.new('RGB', (w, h), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, p in enumerate(imgs):
            im = Image.open(p).convert('RGB').resize((tile, tile), Image.LANCZOS)
            canvas.paste(im, (i*tile, 0))
            name = os.path.basename(p)
            draw.text((i*tile+5, tile+2), name, font=font, fill='black')
        out = os.path.join(out_dir, f"{cat}_recs.png")
        canvas.save(out)

if __name__ == "__main__":
    make_recommendations(
        root='dataset/test',      # pasta de teste com subpastas por categoria
        out_dir='recs',   # onde os PNGs serão salvos
        topk=3,           # número de recomendações
        tile=128          # tamanho de cada miniatura
    )
