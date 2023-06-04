## Environment
- Python 3.7  
- Requirements
    - Install `torch==1.9.0` according to your CUDA environment.
    - ```pip install -r requirements.txt```
    - ```cd global_utils; pip install -e .```
    
## Data preparation
- The GloVe embeddings can be downloaded from [here](https://nlp.stanford.edu/projects/glove/). 
- The Google news embeddings can be downloaded from [here](https://code.google.com/archive/p/word2vec/). 
- The STS datasets can be found in `./examples/datasets/` in [Huang's repository](https://github.com/Jun-jie-Huang/WhiteningBERT).

## Experimental commands
```
python3 src/run_train.py --corpus_path <PATH> --model_name_or_path bert-base-cased --result_dir <PATH> --lr 0.001 --adv_lr 0.001 --epoch 3 --batch_size 128 --adv_freq_thresh 3000 --adv_lambda 0.1 --save_selected_epoch 1_3 --layer_index 1_12
```
