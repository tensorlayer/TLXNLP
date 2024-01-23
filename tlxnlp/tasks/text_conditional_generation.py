import time

import tensorlayerx as tlx
from tqdm import tqdm

from ..metrics import bleu


class TextForConditionalGeneration(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(TextForConditionalGeneration, self).__init__()
        self.backbone = backbone

        self.model_dim = self.backbone.model_dim
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", True)

        if not self.tie_word_embeddings:
            initializer_factor = kwargs.pop(
                "initializer_factor", self.backbone.initializer_factor
            )
            vocab_size = kwargs.pop("vocab_size", self.backbone.vocab_size)
            lm_head_initializer = tlx.initializers.RandomNormal(
                mean=0, stddev=initializer_factor
            )
            self.lm_head = tlx.layers.Dense(
                vocab_size, b_init=None, name="lm_head", W_init=lm_head_initializer
            )

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.backbone.decoder_start_token_id
        pad_token_id = self.backbone.pad_token_id

        start_tokens = decoder_start_token_id * tlx.ones((input_ids.shape[0], 1))
        start_tokens = tlx.cast(
            start_tokens, input_ids.dtype
        )  # Ensure compatible dtypes for concatenation
        shifted_input_ids = tlx.concat([start_tokens, input_ids[:, :-1]], -1)

        assert pad_token_id is not None, "pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = tlx.where(
            shifted_input_ids == -100,
            tlx.cast(
                pad_token_id * tlx.ones(shifted_input_ids.shape),
                shifted_input_ids.dtype,
            ),
            shifted_input_ids,
        )

        shifted_input_ids = tlx.identity(shifted_input_ids)

        return shifted_input_ids

    def loss_fn(self, logits, labels):
        loss = tlx.losses.cross_entropy_seq_with_mask

        mask = tlx.not_equal(labels, -100)
        logits = tlx.reshape(logits, shape=(-1, self.backbone.vocab_size))
        labels = tlx.where(mask, labels, 0)
        return loss(logits=logits, target_seqs=labels, input_mask=mask)

    def generate(self, inputs=None, attention_mask=None):
        ...

    def generate_one(self, inputs=None, attention_mask=None):
        decoder_start_token_id = self.backbone.decoder_start_token_id
        start_tokens = decoder_start_token_id * tlx.ones((inputs.shape[0], 1))
        start_tokens = tlx.cast(start_tokens, inputs.dtype)

        max_length = 64
        encoder_outputs = self.backbone.encoder(
            inputs,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
        )

        while (
            int(start_tokens[0][-1]) != self.backbone.eos_token_id
            and start_tokens.shape[1] < max_length
        ):
            logits = self.forward(
                inputs=inputs,
                attention_mask=attention_mask,
                decoder_input_ids=start_tokens,
                encoder_outputs=encoder_outputs,
            )
            last_tokens = tlx.argmax(logits, -1)[:, -1:]
            start_tokens = tlx.concat([start_tokens, last_tokens], axis=-1)
        return start_tokens

    def forward(
        self,
        inputs=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.backbone(
            inputs=inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        sequence_output = decoder_outputs[0]

        if self.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
            logits = self.backbone.shared(sequence_output, mode="linear")
        else:
            logits = self.lm_head(sequence_output)

        logits = tlx.cast(logits, tlx.float32)

        return logits


def valid_bleu(model, test_dataset, transform_fn):
    model.set_eval()
    targets = []
    predictions = []
    for X_batch, y_batch in tqdm(test_dataset, total=float("inf")):
        decode_id = model.generate_one(
            inputs=X_batch["inputs"], attention_mask=X_batch["attention_mask"]
        )
        decode_str = transform_fn(decode_id[0])
        label_str = transform_fn(y_batch["labels"][0])
        targets.append(label_str)
        predictions.append(decode_str)

    print(bleu(targets, predictions))


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
                    metrics.update(_logits, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                n_iter += 1

                if print_train_batch:
                    print(
                        "Epoch {} of {} {} took {}".format(
                            epoch + 1, n_epoch, n_iter, time.time() - start_time
                        )
                    )
                    print("   train loss: {}".format(train_loss / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print(
                    "Epoch {} of {} took {}".format(
                        epoch + 1, n_epoch, time.time() - start_time
                    )
                )
                print("   train loss: {}".format(train_loss / n_iter))
        if test_dataset is not None:
            tran_func = test_dataset.dataset.transforms[0].ids_to_string
            valid_bleu(network, test_dataset, tran_func)
