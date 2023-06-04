# -*- coding: utf-8 -*-
import codecs
from utils.utils import get_total_line
import logging
logger = logging.getLogger()


class DataHandler(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.test = args.test
        self.total_line = None
        self.corpus_path = args.corpus_path
        self.tokenizer = tokenizer
        self.train_data = None

    def prepare_dataset(self):
        self.train_data = self.load_dataset()

    def load_dataset(self):
        dataset = []
        self.total_line = get_total_line(path=self.corpus_path, test=self.test)
        logger.info("Create dataset ...")
        with codecs.open(self.corpus_path, "r", 'utf-8', errors='replace') as input_data:
            for i, line in enumerate(input_data):

                if i % int(self.total_line / 10) == 0:
                    logger.info('{} % done'.format(round(i / (self.total_line / 100))))

                sentence = line.rstrip('\n')
                dataset.append((sentence))

                if self.test and len(dataset) == 1000:
                    logger.info("Prepared small dataset for quick test.")
                    break
        logger.info("Create dataset ... done")
        logger.info(f'len(dataset) = {len(dataset)}')
        return dataset


