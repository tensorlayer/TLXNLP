import os

# os.environ["TL_BACKEND"] = "torch"
os.environ["TL_BACKEND"] = "paddle"
# os.environ["TL_BACKEND"] = "tensorflow"

from tlxnlp.tasks.text_classification import TextClassification
from tlxnlp.models.transform import BasicEmbedding
from tlxnlp.models import Rnn, Lstm, TextCNN
import tensorlayerx as tlx

if __name__ == "__main__":
    tk_emb = BasicEmbedding()
    backbone = TextCNN(vocab_size=tk_emb.get_vocab_size())
    model = TextClassification(backbone)
    model.load_weights("./demo/text_classification/model_cnn.npz")
    model.set_eval()

    text = "it 's a charming and often affecting journey ."
    x, _ = tk_emb(text, None)
    x = tlx.expand_dims(x, 0)

    _logits = model(x)
    label = tlx.argmax(_logits, axis=-1)
    print(label)
