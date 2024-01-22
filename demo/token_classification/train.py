import os

# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

from tlxnlp.datasets import Conll2003
from tlxnlp.tasks.text_token_classification import TextTokenClassification, Trainer
from tlxnlp.models.transform import BertTransform
from tlxnlp.models import Bert
from tensorlayerx.dataflow import DataLoader
import tensorlayerx as tlx

if __name__ == "__main__":
    datas = Conll2003.load("./data/conll2003", task="ner")
    train_dataset = datas["train"]
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataset = datas["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    transform = BertTransform(
        vocab_file="./demo/text_classification/vocab.txt", task="token", max_length=128
    )
    train_dataset.register_transform_hook(transform)
    test_dataset.register_transform_hook(transform)

    backbone = Bert()
    model = TextTokenClassification(backbone, n_class=9)
    optimizer = tlx.optimizers.Adam(lr=0.00001)
    metric = tlx.metrics.Accuracy()
    trainer = Trainer(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metric
    )
    trainer.train(
        n_epoch=1,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=True,
    )

    model.save_weights("./demo/token_classification/model.npz")
