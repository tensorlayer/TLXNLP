import os

# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

from tlxnlp.datasets import SST2
from tlxnlp.tasks.text_classification import TextClassification, Trainer
from tlxnlp.models.transform import BertTransform
from tlxnlp.models import Bert
from tensorlayerx.dataflow import DataLoader
import tensorlayerx as tlx

if __name__ == "__main__":
    datas = SST2.load("./data/sst2")
    train_dataset = datas["train"]
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataset = datas["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    transform = BertTransform(
        vocab_file="./demo/text_classification/vocab.txt", max_length=128
    )
    train_dataset.register_transform_hook(transform)
    test_dataset.register_transform_hook(transform)

    backbone = Bert()
    model = TextClassification(backbone)
    optimizer = tlx.optimizers.Adam(lr=0.00001)
    loss_fn = model.loss_fn
    metric = tlx.metrics.Accuracy()
    model = Trainer(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
    model.train(
        n_epoch=1,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=True,
    )

    model.save_weights("./demo/text_classification/model.npz")
