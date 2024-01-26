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
            logits = self.forward({
                "inputs": inputs,
                "attention_mask": attention_mask,
                "decoder_input_ids": start_tokens,
                "encoder_outputs": encoder_outputs
            })
            last_tokens = tlx.argmax(logits, -1)[:, -1:]
            start_tokens = tlx.concat([start_tokens, last_tokens], axis=-1)
        return start_tokens

    def forward(self, x):
        if (
            "labels" in x
            and "decoder_input_ids" not in x
            and "decoder_inputs_embeds" not in x
        ):
            # get decoder inputs from shifting lm labels to the right
            x["decoder_input_ids"] = self._shift_right(x["labels"])

        decoder_outputs = self.backbone(x)
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
    for X_batch, y_batch in tqdm(test_dataset):
        decode_id = model.generate_one(
            inputs=X_batch["inputs"], attention_mask=X_batch["attention_mask"]
        )
        decode_str = transform_fn(decode_id[0])
        label_str = transform_fn(y_batch[0])
        targets.append(label_str)
        predictions.append(decode_str)

    print(bleu(targets, predictions))
