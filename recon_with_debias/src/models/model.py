from torch import nn


class ReconModel(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(ReconModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,
                                      embed_dim)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.normal_(mean=0, std=0.01)

    def forward(self, idx):
        embeddings = self.embedding(idx)
        return embeddings
