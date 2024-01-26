import os

# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

from tlxnlp.tasks.text_classification import TextClassification
from tlxnlp.models.transform import BertTransform
from tlxnlp.models import Bert
import tensorlayerx as tlx

if __name__ == "__main__":
    backbone = Bert()
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
