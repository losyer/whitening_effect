import logging
import numpy as np

logger = logging.getLogger(__name__)


def whitening_torch(embeddings):
    import torch
    embeddings = torch.tensor(embeddings)
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings.detach().numpy()


class Vectors(object):
    def __init__(self, args, max_num_words=None
                 ):
        self.args = args
        self.word2idx = {}
        self.vectors = None
        self.max_num_words = max_num_words

    def load_vectors(self, vector_path, total_line=2000000):
        import codecs
        logger.info("Loading vectors ...")
        vectors = []
        with codecs.open(vector_path, "r", 'utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i % int(total_line / 10) == 0:
                    logger.info('{} % done'.format(round(i / (total_line / 100))))
                # col = line.strip().split()
                if i == 0:
                    col = line.strip('\n').split()
                    vocab_size, dim = int(col[0]), int(col[1])
                    continue
                col = line.rstrip(' \n').rsplit(' ', dim)
                word = col[0]
                self.word2idx[word] = i - 1
                vector = np.array(col[1:], dtype=np.float32)
                try:
                    assert len(vector) == dim
                except:
                    print(f'Error: word = {word}')
                    continue
                vectors.append(vector)

                if self.max_num_words:
                    if len(vectors) == self.max_num_words:
                        break

        logger.info("Loading vectors ... done")
        logger.info(f'len(vectors) = {len(vectors)}')
        self.vectors = np.array(vectors, dtype=np.float32)

    def word_vec(self, word):
        idx = self.word2idx[word]
        vec = self.vectors[idx]
        return vec


def cal_similarity(sent_reps, dim=300):
    from sklearn.metrics.pairwise import cosine_similarity
    scores = []
    for rep1, rep2 in sent_reps:
        score = cosine_similarity(np.reshape(rep1, (1, dim)),
                                  np.reshape(rep2, (1, dim)))
        score = score[0][0]
        scores.append(score)
    return scores


def sper_corrcoef(targets, predictions):
    import scipy as sc
    return 100 * sc.stats.spearmanr(targets, predictions)[0] # Outputted nan
    # return 100 * sc.stats.mstats.spearmanr(targets, predictions)[0]
