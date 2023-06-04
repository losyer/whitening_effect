# coding: utf-8
import numpy as np
import json
from tqdm import tqdm
import logging
from nltk.tokenize import word_tokenize
from utils.util import whitening_torch
from utils.util import sper_corrcoef
from utils.util import cal_similarity
from sts import prepare_sentence_to_vec
from sts import prepare_test_set_df


logger = logging.getLogger(__name__)


def load_data(args):
    logger.info('Loading dataset ...')
    dataset_name = [
                    'STS-14',
                    ]
    paths = [
        args.sts_file_path,
    ]
    word_rank_data = [
        args.word_rank_file_path,
    ]
    test_dfs = []
    for path in paths:
        test_dfs.append(prepare_test_set_df(path))

    return test_dfs, dataset_name, word_rank_data


def main(args):
    test_dfs, dataset_name, word_rank_data = load_data(args)
    sentence_to_vec = prepare_sentence_to_vec(args, args.vector_type)

    dim = args.dim
    for i, test_df in enumerate(test_dfs):
        data_name = dataset_name[i]
        logger.info(f'For dataset {data_name}')
        logger.info('Getting representations ...')
        sent_reps = []
        for j in tqdm(range(len(test_df))):
            sentence1 = test_df['sentence1'].iloc[j]
            if args.tokenize:
                sentence1 = ' '.join(word for word in word_tokenize(sentence1))
            if args.lower:
                sentence1 = sentence1.lower()
            vec1 = sentence_to_vec(sentence1)

            sentence2 = test_df['sentence2'].iloc[j]
            if args.tokenize:
                sentence2 = ' '.join(word for word in word_tokenize(sentence2))
            if args.lower:
                sentence2 = sentence2.lower()
            vec2 = sentence_to_vec(sentence2)

            sent_reps.append((vec1, vec2))

        if args.vector_type == 'contextualized-whitening':
            sent_reps_paired = np.array(sent_reps)
            sent_reps_flatten = np.reshape(sent_reps_paired, (len(sent_reps)*2, dim))
            whitened = whitening_torch(sent_reps_flatten)
            sent_reps = np.reshape(whitened, (len(sent_reps), 2, dim))

        #
        # Calculating similarity
        #
        logger.info('Predicting similarity scores ...')
        prediction_score_list = cal_similarity(sent_reps, dim)
        target_score_list = [float(score) for score in test_df['score'].to_list()]
        assert len(prediction_score_list) == len(target_score_list)
        result = sper_corrcoef(target_score_list, prediction_score_list)
        print("Spearman Correlation (before sort): ", result)

        #
        # Sorting data
        #
        with open(word_rank_data[i], 'r') as fi:
            word_ranks = json.load(fi)
        data_list = []
        for j in range(len(word_ranks)):
            t_score = target_score_list[j]
            p_score = prediction_score_list[j]
            word_rank = word_ranks[j]
            data_dict = {'t_score': t_score,
                         'p_score': p_score,
                         'word_rank': word_rank
            }
            data_list.append(data_dict)
        sorted_data = sorted(data_list, key=lambda x: x['word_rank'])

        new_t_scores = []
        new_p_scores = []
        new_word_ranks = []
        for data_dict in sorted_data:
            new_t_scores.append(data_dict['t_score'])
            new_p_scores.append(data_dict['p_score'])
            new_word_ranks.append(data_dict['word_rank'])
        new_p_scores = [float(v) for v in new_p_scores]

        #
        # Evaluation with cutting
        #
        results = []
        avg_word_ranks = []
        cut_length = args.cut_length
        for j in range(len(word_ranks)):
            length = len(word_ranks) - j
            if length > cut_length - 1:
                result = sper_corrcoef(new_t_scores[:length], new_p_scores[:length])
                avg_word_rank = sum(new_word_ranks[:length]) / float(len(new_word_ranks[:length]))
                avg_word_ranks.append(avg_word_rank)
                results.append(result)

        result_dict = {'t_score': new_t_scores,
                       'p_score': new_p_scores,
                       'word_ranks': new_word_ranks,
                       'results': results,
                       'avg_word_ranks': avg_word_ranks}

        name = f'{args.result_dir}/{dataset_name[i]}_{args.out_name}.json'
        with open(name, 'w') as fo:
            json.dump(result_dict, fo)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--rank', type=int, default=0)

    # Paths
    parser.add_argument('--vector_path', type=str)
    parser.add_argument('--result_dir', type=str)

    parser.add_argument('--sts_file_path', type=str)
    parser.add_argument('--word_rank_file_path', type=str)

    # Model Params
    parser.add_argument('--vector_type', type=str)
    parser.add_argument('--dim', type=int, default=300)
    parser.add_argument('--model_name_or_path', type=str, default=None)

    parser.add_argument('--layer_index', type=str, default='12')
    parser.add_argument('--lower', action="store_true")
    parser.add_argument('--tokenize', action="store_true")
    parser.add_argument('--remove_cls_and_sep', action="store_true")
    parser.add_argument('--clip_outlier', type=str, default=None)

    parser.add_argument('--out_name', type=str, default='tmp')
    parser.add_argument('--cut_length', type=int, default=3)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())




