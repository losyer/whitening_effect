import json
import logging
import os
import shutil
from typing import List, Dict, Type
from collections import defaultdict
import transformers
import torch
from torch import nn, Tensor, device
from torch.optim import Optimizer
from tqdm.autonotebook import trange
from matplotlib import pyplot as plt
import torch.nn.functional as F
import global_utils

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, loss_model, w2v_model, dataloader,
                 device: str = None,
                 args=None,
                 epochs: int = 1,
                 scheduler: str = 'WarmupLinear',
                 warmup_steps: int = 10000,
                 optimizer_class: Type[Optimizer] = transformers.AdamW,
                 optimizer_params: Dict[str, object] = {'lr': 1e-4, 'eps': 1e-6, 'correct_bias': False},
                 weight_decay: float = 0.0,
                 evaluation_steps: int = 0,
                 output_path: str = None,
                 save_best_model: bool = True,
                 max_grad_norm: float = 999999.0,
                 use_amp: bool = False,
                 rank: int = 0,
                 save_per_epoch=True,
                 save_selected_epoch: List[int] = [],
                 ):
        self.loss_model = loss_model
        self.w2v_model = w2v_model
        self.dataloader = dataloader
        self.rank = rank
        self.args = args
        self.log = {}

        self.graph_x = []
        self.graph_y_score = []
        self.graph_y_losses = {}
        self.trainer_params = {}

        self.epochs = epochs
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.optimizer_class = optimizer_class
        self.optimizer_params=  optimizer_params
        self.weight_decay = weight_decay
        self.evaluation_steps = evaluation_steps
        self.output_path = output_path
        self.save_best_model = save_best_model
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.rank = rank
        self.save_per_epoch = save_per_epoch
        self.save_selected_epoch = save_selected_epoch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = self.rank

    def run(self):

        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)

        self.loss_model.to(self._target_device)
        steps_per_epoch = len(self.dataloader)

        # Prepare optimizers
        param_optimizer = list(self.loss_model.named_parameters())
        optimizer_class = self.optimizer_class
        optimizer_params = self.optimizer_params
        # Set weight decay
        weight_decay = self.weight_decay
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        ntokens = self.w2v_model.embedding.weight.shape[0]
        rate = (ntokens - self.args.adv_freq_thresh) * 1.0 / ntokens
        adv_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([rate, 1 - rate]).to(self._target_device))
        adv_hidden = nn.Linear(self.args.embed_dim, 2).to(self._target_device)
        adv_hidden.weight.data.uniform_(-0.1, 0.1)
        adv_optimizer = torch.optim.SGD(adv_hidden.parameters(), lr=self.args.adv_lr, weight_decay=self.args.adv_wdecay)
        recon_loss_fn = nn.MSELoss(reduction='none')

        global_step = 0
        epochs = self.epochs
        show_progress_bar = True
        max_grad_norm = self.max_grad_norm
        data_iterator = iter(self.dataloader)
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            self.trainer_params['current_epoch'] = epoch
            training_steps = 0
            loss_sum_dict = defaultdict(float)
            loss_sum_count = 0

            self.loss_model.zero_grad()
            self.loss_model.train()

            for iter_num in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=True):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.dataloader)
                    data = next(data_iterator)

                # features, labels = data
                word_idx, ref_vector, freq, freq_label, y = data
                word_idx = word_idx.to(self._target_device)
                ref_vector = ref_vector.to(self._target_device)
                # freq = freq.to(self._target_device)
                freq_label = freq_label.to(self._target_device)

                vec = self.w2v_model(word_idx)

                # First step
                optimizer.zero_grad()
                adv_optimizer.zero_grad()
                adv_h = adv_hidden(vec)
                adv_loss = adv_criterion(adv_h, freq_label)
                adv_loss.backward()
                adv_optimizer.step()

                # Second step
                optimizer.zero_grad()
                adv_optimizer.zero_grad()
                vec = self.w2v_model(word_idx)

                adv_h = adv_hidden(vec)
                adv_loss = adv_criterion(adv_h, freq_label)

                # Convert to unit length
                if self.args.normalize:
                    vec = F.normalize(vec)
                    ref_vector = F.normalize(ref_vector)
                recon_loss = recon_loss_fn(vec, ref_vector).mean(axis=1)
                recon_loss = recon_loss.mean()

                adv_lambda = self.args.adv_lambda
                loss = recon_loss - adv_lambda * adv_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.w2v_model.parameters(), max_grad_norm)
                optimizer.step()

                # Logging recon loss
                loss_sum_dict['recon_loss'] += float(recon_loss.data.cpu().numpy())
                loss_sum_dict['adv_loss'] += float(adv_loss.data.cpu().numpy())
                loss_sum_count += 1

                # if not skip_scheduler:
                #     scheduler.step()
                training_steps += 1
                global_step += 1

            if self.rank == 0:
                self._eval_during_training(None, self.output_path, self.save_best_model, epoch, -1,
                                           loss_sum_dict, loss_sum_count, global_step,
                                           epoch_end=True, save_selected_epoch=self.save_selected_epoch)

        if self.rank == 0:
            self._save_graph(self.output_path)
            output_path_last = self.output_path + f'/last'
            self.save(output_path_last)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps,
                              loss_sum_dict, loss_sum_count, global_step,
                              epoch_end, save_selected_epoch=[]):
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            self.log['score'] = score
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    output_path_best = output_path + f'/best_epoch-{epoch}_step-{steps}'
                    self.save(output_path_best)
            self.graph_y_score.append(score)

        self.graph_x.append(global_step)
        self.log['global_step'] = global_step

        for loss_name, loss in loss_sum_dict.items():
            loss_avg = loss / loss_sum_count
            if loss_name not in self.graph_y_losses:
                self.graph_y_losses[loss_name] = []
            self.graph_y_losses[loss_name].append(loss_avg)
            self.log[loss_name] = loss_avg
        self.output_log(output_path)

        if epoch_end:
            if epoch in save_selected_epoch:
                output_path_epoch = output_path + f'/selected_epoch_{epoch+1}'
                self.save(output_path_epoch)
            else:
                if self.save_per_epoch:
                    output_path_epoch = output_path + f'/epoch_{epoch+1}'
                    self.save(output_path_epoch)

            pre_epoch_path = output_path + f'/epoch_{epoch}'
            if os.path.exists(pre_epoch_path):
                shutil.rmtree(pre_epoch_path)

    def output_log(self, output_path):
        with open(output_path + '/log.txt', 'a') as fo:
            fo.write(json.dumps(self.log, sort_keys=True, indent=4)+'\n')
        self._save_graph(output_path)

    def _save_graph(self, output_path):
        # logger.info('Saving graphs')
        fig = plt.figure()
        ax1 = fig.subplots()

        colors = global_utils.colors

        color_num = 0
        for loss_name, loss in self.graph_y_losses.items():
            ax1.plot(self.graph_x,
                     self.graph_y_losses[loss_name],
                     c=colors[color_num],
                     label=loss_name)
            color_num += 1

        ax1.legend()

        title = f"Epoch {self.trainer_params['current_epoch']+1} / {self.epochs}\n" \
            f"Params: {self.args.out_prefix}\n"
        plt.title(title)
        plt.savefig(output_path+'/perf_and_loss.png', bbox_inches='tight')
        plt.close()

    def save(self, path):
        if path is None:
            return
        os.makedirs(path, exist_ok=True)
        logger.info("Save model to {}".format(path))

        name = '/word_embeddings.bin'
        torch.save(self.w2v_model.state_dict(), path+name)

