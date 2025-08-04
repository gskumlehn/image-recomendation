import os
import pandas as pd
import requests
from tqdm import tqdm
from urllib.parse import urlparse

def list_csv_files(directory):
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith('.csv')
    ]

def download_image(url, save_dir, prefix, index):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1]
    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
        ext = '.jpg'
    filename = f"{prefix}_{index}{ext}"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as file:
        file.write(response.content)

def process_csv_file(csv_path, dataset_root):
    category = os.path.splitext(os.path.basename(csv_path))[0]
    category_dir = os.path.join(dataset_root, category)
    os.makedirs(category_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    url_column = 'image'
    for index, url in tqdm(enumerate(df[url_column]), total=len(df), desc=category):
        try:
            download_image(url, category_dir, category, index)
        except Exception as error:
            print(f"Error downloading {url}: {error}")

databases_dir = 'databases'
dataset_root = 'dataset/raw/'
csv_files = list_csv_files(databases_dir)
for csv_file in csv_files:
    process_csv_file(csv_file, dataset_root)
