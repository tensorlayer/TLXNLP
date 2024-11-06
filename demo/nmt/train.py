import os
import sys
sys.path.append('../../')

import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader

from tlxnlp.datasets import Text2TextData
from tlxnlp.models import T5Model
from tlxnlp.models.transform import T5Transform
from tlxnlp.tasks.text_conditional_generation import TextForConditionalGeneration, valid_bleu


class EmptyMetric(object):
    def __init__(self):
        return

    def update(self, *args):
        return

    def result(self):
        return 0.0

    def reset(self):
        return


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
    datas = Text2TextData.load("./data/t2t", train_limit=18)
    transform = T5Transform(
        vocab_file="./demo/text_classification/spiece.model",
        source_max_length=128,
        label_max_length=128,
    )
    train_dataset = datas["train"]
    train_dataloader = DataLoader(train_dataset, batch_size=4)
    test_dataset = datas["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    train_dataset.register_transform_hook(transform)
    test_dataset.register_transform_hook(transform)

    backbone = T5Model()
    model = TextForConditionalGeneration(backbone)
    optimizer = tlx.optimizers.Adam(lr=0.0001)
    trainer = tlx.model.Model(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=EmptyMetric())
    trainer.train(
        n_epoch=1,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=True,
    )
    model.save_weights("./demo/nmt/model.npz")

    valid_bleu(model, test_dataloader, transform.ids_to_string)
