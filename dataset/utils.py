import os
import shutil
from pathlib import Path

import requests
from tqdm import tqdm
import zipfile


def download_dataset(url: str,
                     download_dir: Path
                     ) -> Path:
    """
    Downloads file if it doesn't already exists
    :param url:
    :param download_dir: the download dir
    :return:
    """
    filename = url.split("/")[-1]
    os.makedirs(download_dir, exist_ok=True)
    filepath = download_dir / filename
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
    files_already_extracted = os.listdir(folder)
    if len(files_already_extracted) > 0:
        return

    fz = zipfile.ZipFile(src_file, "r")

    root_paths = set()
    files_extracted = []

    for file in tqdm(iterable=fz.namelist(), total=len(fz.namelist())):
        fz.extract(file, str(folder))
        files_extracted.append(file)

        file_path = Path(file)
        root_path = file_path.parent
        if root_path != Path('.'):
            root_paths.add(root_path)

    if len(root_paths) == 1:
        root_path = folder / next(iter(root_paths))
        files = os.listdir(root_path)
        for file in files:
            shutil.move(str(root_path / file), str(folder))

        shutil.rmtree(str(root_path))

    if delete:
        os.remove(src_file)
