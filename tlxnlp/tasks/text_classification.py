import time
import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tqdm import tqdm, trange


class TextClassification(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(TextClassification, self).__init__()
        self.backbone = backbone

        n_class = kwargs.pop("n_class", 2)
        self.method = kwargs.pop("method", "mean")

        initializer = tlx.initializers.RandomNormal(mean=0, stddev=1.0)
        self.classifier = tlx.layers.Linear(
            out_features=n_class, in_features=self.backbone.d_model, W_init=initializer
        )

    def loss_fn(self, logits, labels):
        loss = tlx.losses.softmax_cross_entropy_with_logits(logits, labels)
        return loss

    def forward(self, inputs=None, attention_mask=None, **kwargs):
        hidden_states = self.backbone(
            inputs=inputs, attention_mask=attention_mask, **kwargs
        )

        if self.method == "mean":
            hidden_state = tlx.reduce_mean(hidden_states, axis=1)
            logits = self.classifier(hidden_state)
        elif self.method == "first":
            hidden_state = hidden_states[:, 0]
            logits = self.classifier(hidden_state)
        else:
            hidden_state = tlx.reduce_mean(hidden_states, axis=1)
            logits = self.classifier(hidden_state)

        return logits


def valid(model, test_dataset):
    model.set_eval()
    metrics = tlx.metrics.Accuracy()
    val_acc = 0
    for n_iter, (X_batch, y_batch) in enumerate(
        tqdm(test_dataset, desc="Test batch", leave=False), start=1
    ):
        _logits = model(**X_batch)
        metrics.update(_logits, y_batch["labels"])
        val_acc += metrics.result()
        metrics.reset()

    print('\n', val_acc / n_iter)


class Trainer(tlx.model.Model):
    def tf_train(
        self,
        n_epoch,
        train_dataset,
        network,
        loss_fn,
        train_weights,
        optimizer,
        metrics,
        print_train_batch,
        print_freq,
        test_dataset,
    ):
        import tensorflow as tf

        for epoch in trange(n_epoch, desc="Train epoch", leave=False):
            start_time = time.time()

            train_loss, train_acc = 0, 0
            for n_iter, (X_batch, y_batch) in enumerate(
                tqdm(train_dataset, desc="Train batch", leave=False), start=1
            ):
                network.set_train()

                with tf.GradientTape() as tape:
                    _logits = network(**X_batch)
                    _loss_ce = loss_fn(_logits, **y_batch)
                grad = tape.gradient(_loss_ce, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                train_loss += _loss_ce
                if metrics:
                    metrics.update(_logits, y_batch["labels"])
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))

                if print_train_batch:
                    print()
                    print(
                        "Epoch {} of {} {} took {}".format(
                            epoch + 1, n_epoch, n_iter, time.time() - start_time
                        )
                    )
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print()
                print(
                    "Epoch {} of {} took {}".format(
                        epoch + 1, n_epoch, time.time() - start_time
                    )
                )
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                valid(network, test_dataset)
