import tensorlayerx.nn as nn


class Rnn(nn.Module):

    def __init__(self, vocab_size, embedding_size=256):
        super(Rnn, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_size
        )
        self.rnn = nn.RNN(256, 256, bias=True, batch_first=True)
        # self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=2,
        #     bias=True, batch_first=True)
        self.d_model = embedding_size
        # self.linear = nn.Linear(in_features=256, out_features=2)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        output, hidden = self.rnn(emb)
        # x = tensorlayerx.reduce_mean(output, axis=1)
        return output


def RNN(vocab_size, pretrained, arg, **kwargs):
    model = Rnn(vocab_size=vocab_size, **kwargs)
    # if pretrained:
    #     model = restore_model_nlp(model, arg, model_urls)
    return model


def rnn(vocab_size, pretrained=False, **kwargs):
    return RNN(vocab_size, pretrained, "rnn", **kwargs)
