import os
import tarfile
import urllib.request
#from pathlib import Path
from utils import add_noise_snr

DATA_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
DATA_DIR =""#DEFINE it according to your env!
ARCHIVE_PATH = DATA_DIR / "train-clean-100.tar.gz"
EXTRACTED_PATH = DATA_DIR / "LibriSpeech" / "train-clean-100"

DATA_DIR.mkdir(parents=True, exist_ok=True)
#Extract data
if not ARCHIVE_PATH.exists():
    print("Downloading LibriSpeech...")
    urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH)
else:
    print("Archive already downloaded.")
if not EXTRACTED_PATH.exists():
    print("Extracting archive...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
else:
    print("Data already extracted.")
#Sanity check — print total number of .flac files
flac_files = list(EXTRACTED_PATH.rglob("*.flac"))
print(f"Total .flac files found: {len(flac_files)}")
print("Example file:", flac_files[0])
#Mark extracted folder for output dataset
# (This creates a persistent dataset for the second notebook)
import shutil

OUTPUT_DIR = "" #change according to your env 
if not os.path.exists(OUTPUT_DIR):
    shutil.copytree(EXTRACTED_PATH, OUTPUT_DIR)
