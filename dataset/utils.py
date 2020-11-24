import os
from pathlib import Path

import requests
from tqdm import tqdm
import zipfile


def maybe_download(url: str,
                   directory: Path
                   ) -> Path:
    """
    Downloads file if it doesn't already exists
    :param url:
    :param directory:
    :return:
    """
    filename = url.split("/")[-1]
    os.makedirs(directory, exist_ok=True)
    filepath = directory / filename
    if not os.path.exists(filepath):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filepath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
            progress_bar.close()
    else:
        print("File {} already downloaded".format(filepath))

    return filepath


def unzip_file(src_file: Path,
               folder: Path,
               delete: bool = True
               ):
    """ Unzip file
        Args:
            src_file (Path): Zip file.
            folder (Path): Destination folder.
            delete (bool): Whether or not to delete the zip file after unzipping.
        """
    fz = zipfile.ZipFile(src_file, "r")

    for file in tqdm(iterable=fz.namelist(), total=len(fz.namelist())):
        fz.extract(file, folder)

    if delete:
        os.remove(src_file)
