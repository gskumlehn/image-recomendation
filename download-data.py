# download_dataset.py
import os
import subprocess

# Set up Kaggle API credentials
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    f.write('{"username":"your_kaggle_username","key":"your_kaggle_key"}')
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# Download the dataset
subprocess.run(["kaggle", "datasets", "download", "-d", "paramaggarwal/fashion-product-images-small"])

# Unzip the dataset
subprocess.run(["unzip", "fashion-product-images-small.zip"])