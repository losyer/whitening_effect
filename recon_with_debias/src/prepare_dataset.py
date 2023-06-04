# -*- coding: utf-8 -*-
import codecs
import numpy as np
from utils.utils import get_total_line
import torch
import logging

logger = logging.getLogger()


class DataHandler(object):
    def __init__(self, args):
        self.args = args
        self.ref_vec_path = args.ref_vec_path
        self.test = args.test
        if not self.args.inference and self.args.freq_path != "":
            self.create_word_to_freq_dic()
        self.filtering_words = self.load_filtering_words(args.filtering_words_path)
        self.loaded_words_set = set()
        self.total_line = None

    def get_freq(self, word):
        try:
            return self.word_to_freq_dic[word]
        except:
            return 2

    def create_word_to_freq_dic(self):
        logger.info("Create word to frequency dictionary ...")
        self.word_to_freq_dic = {}
        for line in codecs.open(self.args.freq_path, "r", 'utf-8', errors='replace'):
            col = line.strip().split("\t")
            assert len(col) == 2
            word, freq = col[0], int(col[1])
            self.word_to_freq_dic[word] = freq
        logger.info("Create word to frequency dictionary ... done")

    def load_filtering_words(self, path):
        if path == '':
            return set()
        else:
            logger.info('Loading filtering words ...')
            words = set()
            for line in codecs.open(path, "r", 'utf-8', errors='replace'):
                word = line.strip()
                words.add(word)
            logger.info('Loading filtering words ... done')

            return words

    def prepare_dataset(self):
        logger.info("Create dataset ...")
        self.train_data = self.load_dataset()

        # Optional Handling
        # pass

        logger.info("Create dataset ... done")

    def load_dataset(self):
        dataset = []
        self.total_line = get_total_line(path=self.ref_vec_path, test=self.test)
        skipped_word_count = 0
        with codecs.open(self.ref_vec_path, "r", 'utf-8', errors='replace') as input_data:
            for i, line in enumerate(input_data):

                if i % int(self.total_line / 10) == 0:
                    logger.info('{} % done'.format(round(i / (self.total_line / 100))))

                if i == 0:
                    # Get headder info.
                    col = line.strip('\n').split()
                    vocab_size, dim = int(col[0]), int(col[1])
                    continue
                col = line.rstrip(' \n').rsplit(' ', dim)
                word = col[0]

                # if self.args.inference:
                #     raise NotImplementedError
                # else:
                # Skip special conditions
                if word in self.filtering_words \
                        or (len(word) > 30 and self.args.discard_long_word) \
                        or len(col) != dim + 1:
                    print(line)
                    skipped_word_count += 1
                    continue

                self.loaded_words_set.add(word)

                word_idx = i - 1 - skipped_word_count
                word_idx_array = np.array(word_idx, dtype=np.int32)
                ref = [None] if self.args.inference else col[1:]
                ref_vector = np.array(ref, dtype=np.float32)
                assert len(ref_vector) == dim
                y = np.array(0, dtype=np.int32)
                freq = self.get_freq(word) if self.args.freq_path != "" else 1
                freq_array = np.array(freq, dtype=np.float32)
                freq_label = torch.tensor(0).long() if word_idx < self.args.adv_freq_thresh else torch.tensor(1).long()

                dataset = self.set_dataset(dataset, word_idx_array, ref_vector, freq_array, freq_label, y)

                if self.test and len(dataset) == 1000:
                    logger.info("Prepared small dataset for quick test.")
                    break
        logger.info(f'len(dataset) = {len(dataset)}')
        assert len(dataset) == word_idx + 1
        return dataset

    def set_dataset(self, dataset, word_idx, ref_vector, freq_array, freq_label, y):
        dataset.append((word_idx, ref_vector, freq_array, freq_label, y))
        return dataset
