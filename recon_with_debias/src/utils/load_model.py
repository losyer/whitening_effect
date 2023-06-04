# coding: utf-8
import argparse
import codecs
import sys
import torch
from models.model import ReconModel
from tqdm import tqdm
import logging

def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    log_format = '%(asctime)s [%(levelname)-8s] [%(module)s#%(funcName)s %(lineno)d] %(message)s'
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    handler.flush = sys.stdout.flush
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger
logger = init_logger(None)


def model_setup(args, vocab_size=1000):
    model = ReconModel(vocab_size=vocab_size,
                       embed_dim=args.embed_dim)
    return model


def load_words(args, test=False, total_line=2100000):
    words = []
    with codecs.open(args.ref_vec_path, "r", 'utf-8', errors='replace') as input_data:
        for i, line in enumerate(input_data):

            if i % int(total_line / 10) == 0:
                logger.info('{} % done'.format(round(i / (total_line / 100))))

            if i == 0:
                # Get headder info.
                col = line.strip('\n').split()
                vocab_size, dim = int(col[0]), int(col[1])
                continue
            col = line.rstrip(' \n').rsplit(' ', dim)
            if len(col) != dim + 1:
                print("Error line: ")
                print(line)
                continue
            word = col[0]
            words.append(word)

            if test and len(words) == 1000:
                break
    logger.info(f'len(words) = {len(words)}')
    return words


def output_vector(words, vectors, dim, name='vector.txt'):
    vector_file_name = name
    with open(vector_file_name, 'w') as fo:
        # Write header
        fo.write(f'{len(words)} {dim}\n')
        for i, word in tqdm(enumerate(words)):
            vec = vectors[i]

            assert len(vec) == dim
            fo.write(f'{word}')
            for vec_v in vec:
                fo.write(f' {vec_v:.4f}')
            fo.write('\n')


def main_selected_epoch(args, model_path):
    words = load_words(args)
    vocab_size = len(words)

    # Model setup
    w2v_model = model_setup(args, vocab_size)
    w2v_model.load_state_dict(torch.load(model_path))
    embeddings = w2v_model.embedding.weight.detach().numpy()

    name = args.name
    output_vector(words, embeddings, args.embed_dim, name=name)
    print('Done')


def main(args):
    words = load_words(args)
    vocab_size = len(words)

    # Model setup
    w2v_model = model_setup(args, vocab_size)
    w2v_model.load_state_dict(torch.load(args.load_embedding))
    embeddings = w2v_model.embedding.weight.detach().numpy()

    name = args.name
    output_vector(words, embeddings, args.embed_dim, name=name)
    print('Done')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_vec_path', type=str, default="")
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=300)
    parser.add_argument('--load_embedding', type=str, default="")
    parser.add_argument('--name', type=str, default='vector.txt.tmp')
    parser.add_argument('--save_selected_epoch', type=str, default=None)

    parser.add_argument('--test', action='store_true', help='use tiny dataset')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arguments = parse_args()

    if arguments.save_selected_epoch:
        selected_epochs = args.save_selected_epoch.split('_')
        selected_epochs = [int(epoch) for epoch in selected_epochs]

        main_selected_epoch(args, model_path)
    else:
        main(arguments)

