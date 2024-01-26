import tensorlayerx as tlx
import tensorlayerx.nn as nn


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

    def forward(self, x):
        hidden_states = self.backbone(x)

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
