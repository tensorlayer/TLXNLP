import os
import tensorlayerx as tlx

from tlxnlp.models import Bert
from tlxnlp.models.transform import BertTransform
from tlxnlp.tasks.text_token_classification import TextTokenClassification

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
    backbone = Bert()
    model = TextTokenClassification(backbone, n_class=9)
    model.load_weights("./demo/token_classification/model.npz")

    transform = BertTransform(
        vocab_file="./demo/text_classification/vocab.txt", task="token", max_length=128
    )
    tokens = [
        "CRICKET",
        "-",
        "LEICESTERSHIRE",
        "TAKE",
        "OVER",
        "AT",
        "TOP",
        "AFTER",
        "INNINGS",
        "VICTORY",
        ".",
    ]
    labels = ["O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O"]
    x, y = transform(tokens, labels)

    x = {k: tlx.convert_to_tensor([v]) for k, v in x.items()}

    logits = model(x)

    labels = tlx.argmax(logits, axis=-1)
    print(y)
    print(labels)
