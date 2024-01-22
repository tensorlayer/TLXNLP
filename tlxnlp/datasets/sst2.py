import os
import zipfile

from tensorlayerx import logging
from tensorlayerx.files.utils import maybe_download_and_extract

from .text import BaseDataSet, Datasets


def unzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path."""
    logging.info("Unpacking %s to %s" % (gz_path, new_path))
    with zipfile.ZipFile(gz_path, "r") as zFile:
        for fileM in zFile.namelist():
            zFile.extract(fileM, new_path)


def transformer(path):
    texts, labels = [], []
    with open(path, "r") as fp:
        for index, i in enumerate(fp):
            if index == 0:
                continue
            i = i.strip().rsplit(" ", 1)
            texts.append(i[0])
            labels.append(int(i[1]))
    return texts, labels


class SST2(Datasets):
    @classmethod
    def load(cls, path, train_limit=None):
        maybe_download_and_extract(
            "SST-2.zip", path, "https://dl.fbaipublicfiles.com/glue/data/"
        )
        unzip_file(os.path.join(path, "SST-2.zip"), path)

        train_texts, train_labels = transformer(os.path.join(path, "./SST-2/train.tsv"))
        test_texts, test_labels = transformer(os.path.join(path, "./SST-2/dev.tsv"))

        return cls(
            {
                "train": BaseDataSet(train_texts, train_labels),
                "test": BaseDataSet(test_texts, test_labels),
            }
        )
