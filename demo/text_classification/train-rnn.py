import os
import sys
sys.path.append('../../')
# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

from tlxnlp.datasets import SST2
from tlxnlp.tasks.text_classification import TextClassification
from tlxnlp.models.transform import BertTransform
from tlxnlp.models import Rnn
from tensorlayerx.dataflow import DataLoader
import tensorlayerx as tlx

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
    datas = SST2.load("/home/aistudio-user/userdata/tlxzoo/data/SST-2")
    train_dataset = datas["train"]
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataset = datas["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    transform = BertTransform(
        vocab_file="./demo/text_classification/vocab.txt", max_length=128
    )
    train_dataset.register_transform_hook(transform)
    test_dataset.register_transform_hook(transform)

    backbone = Rnn()
    model = TextClassification(backbone)
    optimizer = tlx.optimizers.Adam(lr=0.00001)
    loss_fn = model.loss_fn
    metric = tlx.metrics.Accuracy()
    model = tlx.model.Model(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
    model.train(
        n_epoch=1,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=False,
    )

    model.save_weights("./demo/text_classification/model.npz")
