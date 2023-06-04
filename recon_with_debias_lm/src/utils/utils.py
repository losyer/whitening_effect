# coding: utf-8
import codecs
import logging
logger = logging.getLogger()


def get_total_line(path, test):
    if not test:
        total_line = 0
        logger.info('get # of lines')
        with codecs.open(path, "r", 'utf-8', errors='replace') as input_data:
            for _ in input_data:
                total_line += 1
        logger.info('done')
        logger.info('# of lines = {}'.format(total_line))
    else:
        total_line = 1000
        logger.info('# of lines = {}'.format(total_line))

    return total_line
