'this script can be used to download the pacs dataset'
import os
import tarfile
from zipfile import ZipFile
import gdown

def stage_path(data_dir, name):
    '''
    creates the path to data_dir/name
    if it does not exist already
    '''
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path

def download_and_extract(url, dst, remove=True):
    '''
    downloads and extracts the data behind the url
    and saves it at dst
    '''
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        with open(dst, "r:gz") as tar:
            tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        with open(dst, "r:") as tar:
            tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zfile = ZipFile(dst, "r")
        zfile.extractall(os.path.dirname(dst))
        zfile.close()

    if remove:
        os.remove(dst)


def download_pacs(data_dir):
    '''
    download and extract dataset pacs.
    Dataset is saved at location data_dir
    '''
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)

if __name__ == '__main__':
    download_pacs('../pacs')
