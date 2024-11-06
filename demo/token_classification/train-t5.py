import os
import sys
sys.path.append('../../')

# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

from tlxnlp.datasets import Conll2003
from tlxnlp.tasks.text_token_classification import TextTokenClassification
from tlxnlp.models.transform import BertTransform
from tlxnlp.models import T5EncoderModel
from tensorlayerx.dataflow import DataLoader
import tensorlayerx as tlx


class TokenClassificationAccuracy(tlx.metrics.Accuracy):
    def update(self, logits, y_batch):
        for logit, y in zip(logits, y_batch):
            mask = tlx.cast(tlx.not_equal(y, -100), dtype=tlx.int32)
            mask_sum = int(tlx.reduce_sum(mask))
            y = y[1 : mask_sum + 1]
            logit = logit[1 : mask_sum + 1]
            super().update(logit, y)


def device_info():
    found = False
    if not found and os.system("npu-smi info > /dev/null 2>&1") == 0:
        cmd = "npu-smi info"
        found = True
    elif not found and os.system("nvidia-smi > /dev/null 2>&1") == 0:
        cmd = "nvidia-smi"
        found = True
    elif not found and os.system("ixsmi > /dev/null 2>&1") == 0:
        cmd = "ixsmi"
        found = True
    elif not found and os.system("cnmon > /dev/null 2>&1") == 0:
        cmd = "cnmon"
        found = True
    
    os.system(cmd)
    cmd = "lscpu"
    os.system(cmd)
    
if __name__ == "__main__":
    device_info()
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

    backbone = T5EncoderModel()
    model = TextTokenClassification(backbone, n_class=9)
    optimizer = tlx.optimizers.Adam(lr=0.00001)
    metric = TokenClassificationAccuracy()
    trainer = tlx.model.Model(
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
