import gzip
import os
import tarfile
import zipfile

from tensorlayerx import logging
from tensorlayerx.dataflow import IterableDataset
from tensorlayerx.files.utils import maybe_download_and_extract

from .text import BaseDataSet, Datasets


def unzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path."""
    logging.info("Unpacking %s to %s" % (gz_path, new_path))
    if tarfile.is_tarfile(gz_path):
        logging.info("Trying to extract tar file")
        tarfile.open(gz_path, "r").extractall(new_path)
        logging.info("... Success!")
    elif zipfile.is_zipfile(gz_path):
        logging.info("Trying to extract zip file")
        with zipfile.ZipFile(gz_path) as zf:
            zf.extractall(new_path)
        logging.info("... Success!")
    elif os.path.splitext(gz_path)[-1].lower() == ".gz":
        logging.info("Trying to extract gz file")
        with gzip.open(gz_path, "rb") as rf:
            src_name = os.path.split(gz_path)[-1]
            new_path += "/" + os.path.splitext(src_name)[0]
            with open(new_path, "wb+") as wf:
                while rf.readable():
                    dat = rf.read(1024 * 1024)
                    wf.write(dat)
        logging.info("... Success!")
    else:
        logging.info(
            "Unknown compression_format only .tar.gz/.tar.bz2/.tar and .zip supported"
        )


class FileDataSet(IterableDataset, BaseDataSet):
    def __init__(self, *args, limit=None, **kwds):
        BaseDataSet.__init__(self, *args, **kwds)
        self.limit = limit

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        yield from self.read(self.data, self.label, self.limit, self.transform)

    def read(self, data, label, limit, transform_fn):
        with open(data) as fd, open(label) as fl:
            for count, (data, label) in enumerate(zip(fd, fl), start=1):
                if limit and count > limit:
                    break
                data = data.strip()
                label = label.strip()
                data, label = transform_fn(data, label)
                yield data, label


class Text2TextData(Datasets):
    @classmethod
    def load(cls, path, train_limit=None):
        tar_file = "training-giga-fren.tar"
        maybe_download_and_extract(
            tar_file, path, "https://www.statmt.org/wmt10/", extract=True
        )
        en_path = os.path.join(path, "giga-fren.release2.fixed.en")
        fr_path = os.path.join(path, "giga-fren.release2.fixed.fr")
        tar_path = os.path.join(path, tar_file)
        if any(not os.path.exists(p + ".gz") for p in (en_path, fr_path)):
            unzip_file(tar_path, path)
        for p in (en_path, fr_path):
            if not os.path.exists(p):
                unzip_file(p + ".gz", path)

        en_test_file = "newstest2014-fren-en.txt"
        fr_test_file = "newstest2014-fren-fr.txt"
        test_url = "https://github.com/tensorlayer/TLXZoo/tree/main/demo/text/nmt/t5/"
        for file in (en_test_file, fr_test_file):
            maybe_download_and_extract(file, path, test_url, extract=True)

        src_train_path, dst_train_path = en_path, fr_path
        src_dev_path, dst_dev_path = (
            os.path.join(path, en_test_file),
            os.path.join(path, fr_test_file),
        )
        return cls(
            {
                "train": FileDataSet(src_train_path, dst_train_path, limit=train_limit),
                "test": FileDataSet(src_dev_path, dst_dev_path),
            }
        )
