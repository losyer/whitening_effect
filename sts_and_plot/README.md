# README
## Environment
- Python 3.7
- Requirements
    - Install `torch==1.9.0` according to your CUDA environment.
    - ```pip install -r requirements.txt```
    
## Data preparation
- The GloVe embeddings can be downloaded from [here](https://nlp.stanford.edu/projects/glove/). 
- The Google news embeddings can be downloaded from [here](https://code.google.com/archive/p/word2vec/). 
- The STS datasets can be found in `./examples/datasets/` in [Huang's repository](https://github.com/Jun-jie-Huang/WhiteningBERT).

## Experimental commands
### whitening_effect
```
export VECTOR_PATH=<PATH>
export OUTPUT_PATH=<PATH>
export DATA_PATH=<PATH>
export SCRIPT_DIR=./
./run_pca_plot.sh
```

### STS evaluation
```
export DEV_DATA_PATH=<STS_DEV_FILE_PATH>
export TEST_DATA_PATH=<STS_TEST_FILE_PATH>
```

#### static
```
python sts.py --type static --vector_path <PATH> --dim 300 --result_dir <PATH> --rank 0 --both --lower --dev_data_path ${DEV_DATA_PATH} --test_data_path ${TEST_DATA_PATH}
```

#### contextualized
```
python sts.py --type contextualized --model_name_or_path <PATH> --dim 768 --result_dir <PATH> --rank 0 --both --layer_index 1_12 --lower --dev_data_path ${DEV_DATA_PATH} --test_data_path ${TEST_DATA_PATH}
```

### Plot word rank
```
export sts_file_path=<STS_FILE_PATH>
export word_rank_file_path=resource/STS-14_word_ranks.json
```

#### static
```
python sts_word_rank.py --sts_file_path ${sts_file_path} --word_rank_file_path ${word_rank_file_path} --vector_type static --vector_path <PATH> --dim 300 --lower --result_dir <PATH1> --out_name GloVe 
python sts_word_rank.py --sts_file_path ${sts_file_path} --word_rank_file_path ${word_rank_file_path} --vector_type static-whitening --vector_path <PATH> --dim 300 --lower --result_dir <PATH1> --out_name GloVe-wh
python sts_word_rank.py --sts_file_path ${sts_file_path} --word_rank_file_path ${word_rank_file_path} --vector_type static --vector_path <PATH> --dim 300 --lower --result_dir <PATH1> --out_name GloVe-Fdeb
python sts_word_rank.py --sts_file_path ${sts_file_path} --word_rank_file_path ${word_rank_file_path} --vector_type static-whitening --vector_path <PATH> --dim 300 --lower --result_dir <PATH1> --out_name GloVe-Fdeb-wh
python plot_sts_word_rank.py --emb_type glove --json_dir <PATH1> --out_dir <PATH2>
```

#### contextulaized
```
python sts_word_rank.py --sts_file_path ${sts_file_path} --word_rank_file_path ${word_rank_file_path} --vector_type contextualized --model_name_or_path bert-base-cased --dim 768 --layer_index 1_12 --lower --result_dir <PATH1> --out_name BERT
python sts_word_rank.py --sts_file_path ${sts_file_path} --word_rank_file_path ${word_rank_file_path} --vector_type contextualized-whitening --model_name_or_path bert-base-cased --dim 768 --layer_index 1_12 --lower --result_dir <PATH1> --out_name BERT-wh
python sts_word_rank.py --sts_file_path ${sts_file_path} --word_rank_file_path ${word_rank_file_path} --vector_type contextualized --model_name_or_path <PATH> --dim 768 --layer_index 1_12 --lower --result_dir <PATH1> --out_name BERT-Fdeb
python sts_word_rank.py --sts_file_path ${sts_file_path} --word_rank_file_path ${word_rank_file_path} --vector_type contextualized-whitening --model_name_or_path <PATH> --dim 768 --layer_index 1_12 --lower --result_dir <PATH1> --out_name BERT-Fdeb-wh
python plot_sts_word_rank.py --emb_type bert --json_dir <PATH1> --out_dir <PATH2>
```
    