import os
import gdown
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Resource management
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'Final_Model')
GDRIVE_FOLDER_ID = "1Z89B9B7sl1PJktfjliS52i83PaCQujoX"

def download_resources():
    """Download required resources from Google Drive"""
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    
    # Download folder from Google Drive
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}?usp=sharing"
    gdown.download_folder(url=url, output=RESOURCES_DIR, quiet=False)
    
    print(f"Resources downloaded to {RESOURCES_DIR}")

def load_resources():
    """Load required resources or download if not present"""
    if not os.path.exists(RESOURCES_DIR):
        print("Downloading required resources...")
        download_resources()

if __name__ == "__main__":
    load_resources()
    