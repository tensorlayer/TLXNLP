import tensorlayerx as tlx

from tlxnlp.models import T5Model
from tlxnlp.models.transform import T5Transform
from tlxnlp.tasks import TextForConditionalGeneration

if __name__ == "__main__":
    backbone = T5Model()
    model = TextForConditionalGeneration(backbone)
    model.load_weights("./demo/nmt/model.npz")
    model.set_eval()

    transform = T5Transform(
        vocab_file="./demo/text_classification/spiece.model",
        source_max_length=128,
        label_max_length=128,
    )

    text = "Plane giants often trade blows on technical matters through advertising in the trade press."
    x, _ = transform(text, "")

    inputs = tlx.convert_to_tensor([x["inputs"]], dtype=tlx.int64)
    attention_mask = tlx.convert_to_tensor([x["attention_mask"]], dtype=tlx.int64)
    decode_id = model.generate_one(inputs=inputs, attention_mask=attention_mask)
    decode_str = transform.ids_to_string(decode_id[0])
    print(decode_str)
