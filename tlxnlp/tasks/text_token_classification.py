import time
import numpy as np
import tensorlayerx as tlx


class TextTokenClassification(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(TextTokenClassification, self).__init__()
        self.backbone = backbone
        self.n_class = kwargs.pop("n_class", 9)

        initializer = tlx.initializers.RandomNormal(mean=0, stddev=1.0)
        self.classifier = tlx.layers.Linear(
            out_features=self.n_class,
            in_features=self.backbone.d_model,
            W_init=initializer,
        )

    def loss_fn(self, logits, labels):
        loss = tlx.losses.cross_entropy_seq_with_mask

        mask = tlx.not_equal(labels, -100)
        logits = tlx.reshape(logits, shape=(-1, self.n_class))
        labels = tlx.where(mask, labels, 0)
        return loss(logits=logits, target_seqs=labels, input_mask=mask)

    def forward(self, inputs=None, attention_mask=None, **kwargs):
        hidden_states = self.backbone(
            inputs=inputs, attention_mask=attention_mask, **kwargs
        )
        logits = self.classifier(hidden_states)
        return logits


def valid(model, test_dataset):
    model.set_eval()
    metrics = tlx.metrics.Accuracy()
    val_acc = 0
    for n_iter, (X_batch, y_batch) in enumerate(test_dataset):
        _logits = model(**X_batch)
        for logit, y in zip(_logits, y_batch["labels"]):
            mask = tlx.cast(tlx.not_equal(y, -100), dtype=tlx.int32)
            mask_sum = int(tlx.reduce_sum(mask))
            y = y[1 : mask_sum + 1]
            logit = logit[1 : mask_sum + 1]
            metrics.update(logit, y)
        val_acc += metrics.result()
        metrics.reset()

    print(val_acc / n_iter)


def load_huggingface_tf_weight(mode, weight_path):
    import h5py

    file = h5py.File(weight_path, "r")
    for w in mode.all_weights:
        name = w.name
        coder = name.split("/")[0]
        hf_weight_name = f"{coder}/tf_bert_for_pre_training/" + name
        hf_weight_name = hf_weight_name.replace("query/weights", "query/kernel")
        hf_weight_name = hf_weight_name.replace("key/weights", "key/kernel")
        hf_weight_name = hf_weight_name.replace("value/weights", "value/kernel")
        hf_weight_name = hf_weight_name.replace("dense/weights", "dense/kernel")
        hf_weight_name = hf_weight_name.replace("biases:0", "bias:0")
        if hf_weight_name not in file:
            print(hf_weight_name)
            continue
        w.assign(file[hf_weight_name])
    return mode


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

        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    _logits = network(**X_batch)
                    _loss_ce = loss_fn(_logits, **y_batch)
                grad = tape.gradient(_loss_ce, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                train_loss += _loss_ce
                if metrics:
                    for logit, y in zip(_logits, y_batch["labels"]):
                        mask = tlx.cast(tlx.not_equal(y, -100), dtype=tlx.int32)
                        mask_sum = int(tlx.reduce_sum(mask))
                        y = y[1 : mask_sum + 1]
                        logit = logit[1 : mask_sum + 1]
                        metrics.update(logit, y)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                n_iter += 1

                if print_train_batch:
                    print(
                        "Epoch {} of {} {} took {}".format(
                            epoch + 1, n_epoch, n_iter, time.time() - start_time
                        )
                    )
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print(
                    "Epoch {} of {} took {}".format(
                        epoch + 1, n_epoch, time.time() - start_time
                    )
                )
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                valid(network, test_dataset)
