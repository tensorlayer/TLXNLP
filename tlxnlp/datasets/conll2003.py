import os
import zipfile

from tensorlayerx import logging
from tensorlayerx.files.utils import maybe_download_and_extract

from .text import BaseDataSet, Datasets


def unzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path."""
    logging.info("Unpacking %s to %s" % (gz_path, new_path))
    zFile = zipfile.ZipFile(gz_path, "r")
    for fileM in zFile.namelist():
        zFile.extract(fileM, new_path)
    zFile.close()


def transformer(path, tags_set, task):
    texts, labels = [], []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if len(tokens) != 0:
                    texts.append(tokens)
                    labels.append(tags)
                tokens, tags = [], []
            else:
                splits = line.split("\t")
                tokens.append(splits[0])
                if task == "pos":
                    tags.append(tags_set.index(splits[1]))
                elif task == "chunk":
                    tags.append(tags_set.index(splits[2]))
                else:
                    tags.append(tags_set.index(splits[-1].rstrip()))
    return texts, labels


def read_tags(path, task):
    tags_set = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                ...
            else:
                splits = line.split("\t")
                if task == "pos":
                    tags_set.add(splits[1])
                elif task == "chunk":
                    tags_set.add(splits[2])
                else:
                    tags_set.add(splits[-1].rstrip())
    return list(tags_set)


class Conll2003(Datasets):
    @classmethod
    def load(cls, path, task, train_limit=None):
        maybe_download_and_extract("conll2003.zip", path, "https://data.deepai.org/")
        unzip_file(os.path.join(path, "conll2003.zip"), os.path.join(path, "conll2003"))
        tags_set = read_tags(os.path.join(path, "conll2003/valid.txt"), task)

        train_texts, train_labels = transformer(
            os.path.join(path, "conll2003/train.txt"), tags_set, task
        )
        valid_texts, valid_labels = transformer(
            os.path.join(path, "conll2003/valid.txt"), tags_set, task
        )
        test_texts, test_labels = transformer(
            os.path.join(path, "conll2003/test.txt"), tags_set, task
        )

        return cls(
            {
                "train": BaseDataSet(train_texts, train_labels),
                "val": BaseDataSet(valid_texts, valid_labels),
                "test": BaseDataSet(test_texts, test_labels),
            }
        )
