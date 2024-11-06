import os

# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

from tlxnlp.tasks.text_classification import TextClassification
from tlxnlp.models.transform import BertTransform
from tlxnlp.models import Lstm
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
    backbone = Lstm()
    model = TextClassification(backbone)
    model.load_weights("./demo/text_classification/model.npz")
    model.set_eval()

    transform = BertTransform(
        vocab_file="./demo/text_classification/vocab.txt", max_length=128
    )

    text = "it 's a charming and often affecting journey ."
    x, _ = transform(text, None)
    x = {k: tlx.convert_to_tensor([v]) for k, v in x.items()}

    _logits = model(x)
    label = tlx.argmax(_logits, axis=-1)
    print(label)
