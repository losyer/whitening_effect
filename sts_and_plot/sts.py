# coding: utf-8
import numpy as np
import pandas as pd
import csv
import json
from tqdm import tqdm
import sys
import os
from nltk.tokenize import word_tokenize
from utils.util import whitening_torch
from utils.util import Vectors
from utils.util import sper_corrcoef
from utils.util import cal_similarity
from transformers import (
    AutoTokenizer,
    AutoModel,
)
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
    logger.addHandler(handler)

    return logger


logger = init_logger(None)


def prepare_test_set_df(path):
    # For datasets in https://github.com/Jun-jie-Huang/WhiteningBERT
    column_names = ['score', 'sentence1', 'sentence2']
    contents = []
    for i, line in enumerate(open(path)):
        cols = line.rstrip('\n').split('\t')
        cols = [cols[4], cols[5], cols[6]]
        assert len(cols) == len(column_names)
        contents.append(cols)

    test_df = pd.DataFrame(contents, columns=column_names)
    return test_df


def convert_sentence(sentence, sentence_to_vec, args):
    if args.tokenize:
        sentence = ' '.join(word for word in word_tokenize(sentence))
    if args.lower:
        sentence = sentence.lower()
    vec = sentence_to_vec(sentence)
    return vec


def load_dataset(args):
    logger.info('Loading dataset ...')
    dataset_name = ['STSB-dev',
                    'STSB-test'
                    ]
    paths = [args.dev_data_path,
             args.test_data_path]

    test_dfs = []
    for path in paths:
        test_dfs.append(prepare_test_set_df(path))

    return test_dfs, dataset_name


def prepare_sentence_to_vec(args, vector_type):
    logger.info(f"Preparing embedding model (type={vector_type})")
    if vector_type.startswith('static'):
        if args.test:
            max_num_words = 100000
        else:
            max_num_words = 9999999
        vector_path = args.vector_path
        client = Vectors(args,
                         max_num_words=max_num_words)
        client.load_vectors(vector_path)

        if vector_type == 'static':
            pass
        elif vector_type == 'static-whitening':
            embeddings = client.vectors
            whitened_embeddings = whitening_torch(embeddings)
            client.vectors = whitened_embeddings
        else:
            raise ValueError

        def sentence_to_vec(sentence, dim=300):
            words = word_tokenize(sentence)
            vec = []
            for word in words:
                try:
                    vec.append(client.word_vec(word))
                except:
                    pass
            if vec == []:
                vec = np.zeros(dim)
            else:
                vec = np.array(vec).mean(axis=0)
            return vec

    elif vector_type.startswith('contextualized'):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        rank = args.rank
        if rank != -1:
            model = AutoModel.from_pretrained(args.model_name_or_path,
                                              output_hidden_states=True).to(rank)
        else:
            model = AutoModel.from_pretrained(args.model_name_or_path)
        layer_index = args.layer_index.split('_')
        clip_dim = args.clip_outlier.split('_') if args.clip_outlier else None

        def sentence_to_vec(sentence, dim=768):
            if rank != -1:
                inputs = tokenizer(sentence, return_tensors="pt").to(rank)
            else:
                inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs)
            tokens_reps = outputs['last_hidden_state']
            all_hidden_states = outputs['hidden_states']

            if layer_index == ['12']:
                shape1 = tokens_reps.shape[1]
                shape2 = tokens_reps.shape[2]
                tokens_reps = tokens_reps.reshape((shape1, shape2))
                if args.remove_cls_and_sep:
                    tokens_reps = tokens_reps[1:-1]
                averaged_reps = tokens_reps.mean(axis=0)
            else:
                for i, l_idx in enumerate(layer_index):
                    l_idx = int(l_idx)
                    layer_reps = all_hidden_states[l_idx]
                    shape1 = layer_reps.shape[1]
                    shape2 = layer_reps.shape[2]
                    layer_reps = layer_reps.reshape((shape1, shape2))
                    if args.remove_cls_and_sep:
                        layer_reps = layer_reps[1:-1]
                    averaged_reps = layer_reps.mean(axis=0)
                    if i == 0:
                        tmp_reps = averaged_reps
                    else:
                        tmp_reps += averaged_reps
                else:
                    tmp_reps = tmp_reps / float(len(layer_index))
                    averaged_reps = tmp_reps
            if clip_dim:
                for t_dim in clip_dim:
                    averaged_reps[int(t_dim) - 1] = 0.0

            return averaged_reps.detach().cpu().numpy()
    else:
        raise ValueError

    return sentence_to_vec


def solve_sts(args, vector_type, result_dir_arg=None):
    test_dfs, dataset_name = load_dataset(args)

    sentence_to_vec = prepare_sentence_to_vec(args, vector_type=vector_type)

    dim = args.dim
    result_dict = {}
    for i, test_df in enumerate(test_dfs):
        data_name = dataset_name[i]
        logger.info(f'For dataset {data_name}')
        logger.info('Getting representations ...')
        sent_reps = []
        for j in tqdm(range(len(test_df))):
            sentence1 = test_df['sentence1'].iloc[j]
            sentence2 = test_df['sentence2'].iloc[j]
            vec1 = convert_sentence(sentence1, sentence_to_vec, args)
            vec2 = convert_sentence(sentence2, sentence_to_vec, args)
            sent_reps.append((vec1, vec2))

        if vector_type == 'contextualized-whitening':
            sent_reps_paired = np.array(sent_reps)
            sent_reps_flatten = np.reshape(sent_reps_paired, (len(sent_reps) * 2, dim))
            whitened = whitening_torch(sent_reps_flatten)

            sent_reps = np.reshape(whitened, (len(sent_reps), 2, dim))

        #
        # Evaluation
        #
        logger.info('Predicting similarity scores ...')
        prediction_score_list = cal_similarity(sent_reps, dim)
        target_score_list = [float(score) for score in test_df['score'].to_list()]
        assert len(prediction_score_list) == len(target_score_list)

        result = sper_corrcoef(target_score_list, prediction_score_list)
        print("Spearman Correlation: ", result)
        result_dict[dataset_name[i]] = result

    #
    # Save results
    #
    if result_dir_arg:
        result_dir = result_dir_arg
    else:
        result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "settings.json"), "w") as fo:
        fo.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(result_dir, "results.json"), "w") as fo:
        fo.write(json.dumps(result_dict, sort_keys=True, indent=4))
    with open(os.path.join(result_dir, "results.csv"), "w") as fo:
        writer = csv.writer(fo)

        if not args.save_only_values:
            keys = list(result_dict.keys())
            writer.writerow(keys)

        values = list(result_dict.values())
        writer.writerow(values)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--rank', type=int, default='0')

    parser.add_argument('--type', type=str)
    parser.add_argument('--vector_path', type=str)
    parser.add_argument('--dev_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--result_dir', type=str, default='/tmp/')

    # Params
    parser.add_argument('--dim', type=int, default=300)
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--layer_index', type=str, default='12')

    # Flags
    parser.add_argument('--save_only_values', action="store_true")
    parser.add_argument('--lower', action="store_true")
    parser.add_argument('--tokenize', action="store_true")
    parser.add_argument('--remove_cls_and_sep', action="store_true")
    parser.add_argument('--clip_outlier', type=str, default=None)

    parser.add_argument('--both', action="store_true")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.both:
        # Base
        solve_sts(args,
                  vector_type=f'{args.type}',
                  result_dir_arg=args.result_dir+'/Fdeb/')

        # Whitening
        solve_sts(args,
                  vector_type=f'{args.type}-whitening',
                  result_dir_arg=args.result_dir + '/wh/')
    else:
        solve_sts(args,
                  vector_type=args.type)

