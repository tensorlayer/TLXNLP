import tensorlayerx as tlx

from tlxnlp.models import Bert
from tlxnlp.models.transform import BertTransform
from tlxnlp.tasks.text_token_classification import TextTokenClassification

if __name__ == "__main__":
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

    inputs = tlx.convert_to_tensor([x["inputs"]])
    token_type_ids = tlx.convert_to_tensor([x["token_type_ids"]])
    attention_mask = tlx.convert_to_tensor([x["attention_mask"]])

    logits = model(
        inputs=inputs, token_type_ids=token_type_ids, attention_mask=attention_mask
    )

    labels = tlx.argmax(logits, axis=-1)
    print(y)
    print(labels)
