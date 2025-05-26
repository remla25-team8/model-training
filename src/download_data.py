"""
This module handles downloading data from Google Drive and saving it to a specified directory.
"""

import os
import argparse
import gdown


def download_data(save_dir, file_name):
    """
    Download data from Google Drive and save it to the specified directory.
    Args:
        save_dir (str): The directory to save the downloaded file.
        file_name (str): The name of the file to save the data as.
    """
    os.makedirs(save_dir, exist_ok=True)
    gdrive_url = "https://drive.google.com/uc?id=1ROpZ5jfrpJSp5T3-b8bjZX0MDLDCHMTO"
    gdown.download(gdrive_url, os.path.join(save_dir, file_name), quiet=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=False, default="data/raw")
    parser.add_argument("--file_name", type=str, required=False, default="raw_data.tsv")
    parser.add_argument(
        "--version",
        type=str,
        required=False,
        default=None,
        help="Specify the version of the data to download."
    )

    args = parser.parse_args()

    download_data(args.save_dir, args.file_name)
