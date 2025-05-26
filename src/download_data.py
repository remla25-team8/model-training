import pandas as pd
import gdown
import argparse
import os
def download_data (save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    gdrive_url = "https://drive.google.com/uc?id=1ROpZ5jfrpJSp5T3-b8bjZX0MDLDCHMTO"
    gdown.download(gdrive_url, os.path.join(save_dir, file_name), quiet=False)
    
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--save_dir", type=str, required=False, default="data/raw")
    args.add_argument("--file_name", type=str, required=False, default="raw_data.tsv")
    args.add_argument("--version", type=str, required=False, default=None) #TODO: We need to have versioning for the data

    args = args.parse_args()

    download_data(args.save_dir, args.file_name)
    