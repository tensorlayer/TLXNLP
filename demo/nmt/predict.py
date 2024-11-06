import tensorlayerx as tlx

from tlxnlp.models import T5Model
from tlxnlp.models.transform import T5Transform
from tlxnlp.tasks import TextForConditionalGeneration

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
