import codecs
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm, trange
from collections import defaultdict
import random
import numpy as np
np.random.seed(0)

import torch
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)


def init_logger(name):
    import logging
    import sys
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    log_format = '%(asctime)s [%(levelname)-8s] [%(module)s#%(funcName)s %(lineno)d] %(message)s'
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    handler.flush = sys.stdout.flush
    logger.addHandler(handler)

    return logger


logger = init_logger(None)


def global_pca(representations, n_pc: int, num_of_dim=768):
    mean_vec = representations.mean(axis=0)
    mean_vec_tile = np.tile(mean_vec, (representations.shape[0], 1))
    zero_mean_representation = representations - mean_vec_tile

    model = PCA()
    # post_rep = np.zeros((representations.shape[0], representations.shape[1]))
    model.fit(zero_mean_representation)
    component = np.reshape(model.components_, (-1, num_of_dim))
    n_pc = min(n_pc, model.components_.shape[0])

    post_rep = []
    for pre_vec in zero_mean_representation:
        sum_vec = np.zeros(num_of_dim)
        for j in range(n_pc):
            sum_vec = sum_vec + np.dot(pre_vec, np.transpose(component)[:, j].reshape((num_of_dim, 1))) * component[j]

        post_rep.append(pre_vec - sum_vec)
    post_rep = np.array(post_rep)

    return post_rep


def whitening_torch(embeddings):
    import torch
    embeddings = torch.tensor(embeddings)
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings.detach().numpy()


def batch_encoding(args, sentences, tokenizer, model):
    device = args.device
    all_token_reps = []
    all_attn_masks = []
    all_input_ids = []
    batch_size = args.batch_size
    max_length = args.max_length
    logger.info(f'Max length = {max_length}')
    for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
        sentences_batch = sentences[start_index:start_index + batch_size]
        inputs = tokenizer.batch_encode_plus(sentences_batch,
                                             max_length=max_length,
                                             return_tensors='pt',
                                             truncation=True,
                                             pad_to_max_length=True).to(device)

        with torch.no_grad():
            output = model(**inputs)[0]
            input_ids = inputs['input_ids']
            output_array = output.detach().cpu().numpy()
            input_ids_array = input_ids.detach().cpu().numpy()

            excluded_idx = []
            excluded_idx += [101, 102]
            excluded_idx += [133, 8362, 1377, 135]
            tmp_attn_mask = make_attn_mask(input_ids, exclude_idx=excluded_idx)
            for vecs, attn_mask, input_id in zip(output_array, tmp_attn_mask, input_ids_array):
                # Remove PAD (and CLS, SEP)
                mask_idx = np.where(attn_mask == 0)
                processed_vecs = np.delete(vecs, mask_idx, axis=0)
                one_input_id = np.delete(input_id, mask_idx, axis=0)
                all_token_reps.append(processed_vecs)
                all_attn_masks.append(attn_mask)
                all_input_ids.append(one_input_id)

    return all_token_reps, all_attn_masks, all_input_ids


def load_penn_data(args):
    sentences = []
    for line in open(args.data_path):
        sentences.append(line.rstrip('\n'))

        if args.test and len(sentences) == 1000:
            break
    logger.info(f'# of sentences: {len(sentences)}')
    return sentences


def load_vectors(vector_path, args, total_line=2000000):
    logger.info("Loading vectors ...")
    vectors = []
    words = []
    with codecs.open(vector_path, "r", 'utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i % int(total_line / 10) == 0:
                print('{} % done'.format(round(i / (total_line / 100))))
            # col = line.strip().split()
            if i == 0:
                col = line.strip('\n').split()
                vocab_size, dim = int(col[0]), int(col[1])
                continue
            col = line.rstrip(' \n').rsplit(' ', dim)
            word = col[0]
            words.append(word)
            vector = np.array(col[1:], dtype=np.float32)
            try:
                assert len(vector) == dim
            except:
                print("Error:", end='')
                print(line)
                continue

            vectors.append(vector)

            if args.test and len(vectors) == 1000:
                break

    logger.info("Loading vectors ... done")
    logger.info(f'len(vectors) = {len(vectors)}')
    return np.array(vectors, dtype=np.float32), np.array(words)


def plot(alphas, freqs, output_path, args, name='', s=7,
         xmin=None, xmax=None, ymin=None, ymax=None,
         dpi=500, annotations=[], font_size=20, plot_option_alpha=0.8, cmap=plt.cm.jet):
    logger.info('Saving graphs ...')
    alphas_0, alphas_1 = zip(*alphas)
    alphas_0 = np.array(alphas_0)
    alphas_1 = np.array(alphas_1)

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)

    im = ax.scatter(alphas_0, alphas_1,
                    c=freqs, cmap=cmap, edgecolor='none', s=s,
                    alpha=plot_option_alpha)
    if len(annotations) != 0:
        assert len(annotations) == len(alphas_0)
        anno_num = 500
        anno_idx = ([1] * anno_num + [0] * (len(annotations) - anno_num))
        np.random.shuffle(anno_idx)
        for i, label in enumerate(annotations):
            if anno_idx[i] == 1:
                ax.annotate(label, (alphas_0[i], alphas_1[i]), fontsize=2)
    cbar = fig.colorbar(im, ax=ax, aspect=40, pad=0.08, orientation='vertical')

    cbar.ax.tick_params(labelsize=13)
    if xmin and xmax:
        plt.xlim(xmin, xmax)
    if ymin and ymax:
        plt.ylim(ymin, ymax)

    if args.iso_type == 'base':
        title = 'Before whitening'
    else:
        title = 'After whitening'
    # title = f'{args.iso_type} \n' \
    #         f'size={len(alphas_0)}'

    annotation_name = '_anno' if len(annotations) != 0 else ''
    x_label = r"$\alpha_{1}$"
    y_label = r"$\alpha_{2}$"
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=18)
    plt.tick_params(labelsize=15)

    plt.grid()
    out_name = output_path+f'/{name}{args.vector_type}-{args.iso_type}{annotation_name}.png'
    plt.savefig(out_name, bbox_inches='tight', dpi=dpi)
    plt.clf()


def make_attn_mask(input_ids, exclude_idx=[]):
    exclude_idx_set = set(exclude_idx+[0])
    attn_mask_all = []
    for ids in input_ids.cpu().numpy():
        attn_mask = []
        for id in ids:
            if id in exclude_idx_set:
                attn_mask.append(0)
            else:
                attn_mask.append(1)
        attn_mask_all.append(attn_mask)
    return torch.tensor(attn_mask_all)


def prepare_iso_reps(args, model, tokenizer, sentence1, sentence2=[]):
    words, raw_words, vectors, freqs = make_reps(args,
                                                 model,
                                                 tokenizer,
                                                 sentence1,
                                                 sentence2)

    return words, raw_words, vectors, freqs


def make_reps(args, model, tokenizer, sentences1, sentences2=[]):
    #
    # Get representations
    #
    logger.info('Getting representations ...')
    model.trainable = False
    sentences = sentences1 + sentences2
    token_reps, attn_masks, input_ids = batch_encoding(args,
                                                       sentences,
                                                       tokenizer,
                                                       model)

    words = []
    raw_words = []
    vectors = []
    freqs = []
    registered_words = set()
    assert len(token_reps) == len(input_ids)
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    word_reg_count = defaultdict(int)
    for one_token_reps, one_input_ids in zip(token_reps, input_ids):
        assert len(one_token_reps) == len(one_input_ids)

        for token_rep, input_id in zip(one_token_reps, one_input_ids):
            word = id_to_token[input_id]
            if args.reg_same_word:
                if word in registered_words:
                    raw_words.append(word)

                    word_reg_count[word] += 1
                    reg_word = word + f'@{word_reg_count[word]}'
                    words.append(reg_word)
                    vectors.append(token_rep)
                    freqs.append(input_id)
                else:
                    word_reg_count[word] += 1
                    raw_words.append(word)
                    words.append(word)
                    vectors.append(token_rep)
                    freqs.append(input_id)
                    registered_words.add(word)
            else:
                if word in registered_words:
                    pass
                else:
                    raw_words.append(word)
                    words.append(word)
                    vectors.append(token_rep)
                    freqs.append(input_id)
                    registered_words.add(word)

    logger.info(f'# of words: {len(words)}')
    return np.array(words), np.array(raw_words), np.array(vectors), np.array(freqs)


def pca_and_get_alphas(args, vectors_processed):
    logger.info('Conducting PCA and getting alphas ...')
    model = PCA()
    model.fit(vectors_processed)
    dim = args.dim
    component = np.reshape(model.components_, (-1, dim))

    # n_pc = min(n_pc, model.components_.shape[0])
    n_pc = 2
    alphas = []
    for vec in tqdm(vectors_processed):
        tmp_alphas = []
        for i in range(n_pc):
            alpha = np.dot(vec, np.transpose(component)[:, i].reshape((dim, 1)))
            tmp_alphas.append(alpha)
        alphas.append(np.array(tmp_alphas).reshape([n_pc,]))
    alphas = np.array(alphas)

    return alphas


def change_freq_distribution(freqs, words, raw_words, vectors):
    bin_num = 10
    hist, bins = np.histogram(freqs, bin_num)
    print('pre bins')
    print(bins)
    min_num = min(hist)
    logger.info(f'Min value: {min_num}')

    def check_bin_idx(freq, bins, bin_num):
        bin_idx = -1
        for j in range(bin_num):
            if bins[j] <= freq <= bins[j+1]:
                bin_idx = j
        return bin_idx

    pick_idx = []
    bin_count = defaultdict(int)
    for i, freq in enumerate(freqs):
        bin_idx = check_bin_idx(freq, bins, bin_num)
        bin_count[bin_idx] += 1
        if bin_count[bin_idx] <= min_num:
            pick_idx.append(i)
    pick_idx = np.array(pick_idx)
    new_freqs = freqs[pick_idx]
    # new_alphas = alphas[pick_idx]
    new_words = words[pick_idx]
    new_raw_words = raw_words[pick_idx]
    new_vectors = vectors[pick_idx]

    print('after bins')
    print(np.histogram(new_freqs, bins=bins))
    return new_freqs, new_words, new_raw_words, new_vectors


def main(args):
    #
    # Load/encode vectors
    #
    if args.vector_type == 'static':
        vectors, words = load_vectors(args.vector_path,
                                      args)
    elif args.vector_type == 'contextualized':
        sentence = load_penn_data(args)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModel.from_pretrained(args.model_name_or_path).to(args.device)
        words, raw_words, vectors, freqs = prepare_iso_reps(args,
                                                            model,
                                                            tokenizer,
                                                            sentence)

        freqs, words, raw_words, vectors = change_freq_distribution(freqs,
                                                                    words,
                                                                    raw_words,
                                                                    vectors)
    else:
        raise NotImplementedError

    #
    # Vector processing
    #
    if args.iso_type == 'base':
        vectors_processed = vectors
    elif args.iso_type == 'global_pca':
        vectors_processed = global_pca(vectors,
                                       n_pc=2,
                                       num_of_dim=args.dim)
    elif args.iso_type == 'whitening':
        vectors_processed = whitening_torch(vectors)

    #
    # Conduct PCA and get vectors in principal component directions
    #
    alphas = pca_and_get_alphas(args, vectors_processed)

    selected_tokens = {'<', 'un', '##k', '>'}
    #
    # Preprocessing
    #
    if args.vector_type == 'static':
        # freqs = np.array(list(range(len(vectors)))[::-1])
        freqs = np.array(list(range(len(vectors))))

    else:
        if args.remove_selected_tokens:
            not_selected_tokens_idx = []
            for i, raw_word in enumerate(raw_words):
                if raw_word in selected_tokens:
                    pass
                else:
                    not_selected_tokens_idx.append(i)

            freqs = freqs[not_selected_tokens_idx]
            alphas = alphas[not_selected_tokens_idx]
            words = words[not_selected_tokens_idx]


    #
    # For all plots
    #
    # cmap = plt.cm.jet
    cmap = plt.cm.magma
    # plot(alphas, freqs, args.output_path, args, s=args.plot_size, plot_option_alpha=args.plot_option_alpha, cmap=cmap)
    # plot(alphas, freqs, args.output_path, args, s=args.plot_size, annotations=words, cmap=cmap)

    #
    # For sampled plot
    #
    sample_size = args.sample_size
    assert len(alphas) > sample_size
    if sample_size == -1:
        sample_size = int(len(alphas) / 10)
    sample_ids = np.array(random.sample(range(0, len(freqs)), k=sample_size))
    sampled_freqs = freqs[sample_ids]
    sampled_alphas = alphas[sample_ids]
    # sampled_words = words[sample_ids]
    plot(sampled_alphas, sampled_freqs, args.output_path, args,
         name='sampled_', s=args.plot_size, plot_option_alpha=args.plot_option_alpha, cmap=cmap)
    # plot(sampled_alphas, sampled_freqs, args.output_path, args,
    #      name='sampled_', s=args.plot_size, annotations=sampled_words, cmap=cmap)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--iso_type', type=str, default='base')
    parser.add_argument('--vector_type', type=str, default='')

    # Params
    parser.add_argument('--dim', type=int, default=300)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--sample_size', type=int, default=99999999999)
    parser.add_argument('--plot_size', type=int, default=1)
    parser.add_argument('--plot_option_alpha', type=float, default=0.5)

    # Path
    parser.add_argument('--vector_path', type=str)
    parser.add_argument('--model_name_or_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='')
    # parser.add_argument('--freq_rank_path', type=str, default=None)

    # Flags
    parser.add_argument('--remove_selected_tokens', action="store_true")
    parser.add_argument('--reg_same_word', action="store_true")

    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())

