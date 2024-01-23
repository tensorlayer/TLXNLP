import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader

from tlxnlp.datasets import Text2TextData
from tlxnlp.models import T5Model
from tlxnlp.models.transform import T5Transform
from tlxnlp.tasks.text_conditional_generation import (
    TextForConditionalGeneration,
    Trainer,
)

if __name__ == "__main__":
    datas = Text2TextData.load("./data/t2t")
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
    trainer = Trainer(network=model, loss_fn=model.loss_fn, optimizer=optimizer)
    trainer.train(
        n_epoch=1,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=True,
    )
    model.save_weights("./demo/nmt/model.npz")
