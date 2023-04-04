import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from model.eval_metrics import *
import torch.nn.functional as F

from base import BaseTrainer
from model.model import sim_matrix
from utils import inf_loop

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.data_loader = data_loader
        
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min(len(x) for x in data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.total_batch_sum = sum(x.batch_size for x in self.data_loader)
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_iterations = self.max_samples_per_epoch // self.total_batch_sum + 1
        with tqdm(zip(*self.data_loader), desc=f"Training epoch {epoch}", total=total_iterations) as progress:
            for batch_idx, data_li in enumerate(progress):
                if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                    break
                for dl_idx, data in enumerate(data_li):
                    # then assume we must tokenize the input, e.g. its a string
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                      truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)
                    data['video_aug'] = data['video_aug'].to(self.device)
                    data['label'] = data['label'].to(self.device)

                    self.optimizer.zero_grad()
                    # pred, l_pos_neg, cl_labels = self.model(data, bTrain=True)

                    pred, video_feat, text_feat, temp = self.model(data, bTrain=True)
                    pred_loss = self.criterion(pred, data['label'])

                    # ========== (in-batch) VTC loss ==========
                    b = video_feat.size(0)

                    # add cross-modality loss
                    sim_v2t = video_feat @ text_feat.t() / temp.mean()
                    sim_t2v = text_feat @ video_feat.t() / temp.mean()
                    
                    sim_targets = torch.zeros_like(sim_v2t)
                    sim_targets[:,0:b] = torch.eye(b)

                    loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
                    loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean()

                    #add in-modality g2g loss
                    # sim_v2v = video_feat @ video_feat_aug.t() / temp.mean()
                    # sim_t2t = text_feat @ text_feat_aug.t() / temp.mean()

                    # loss_v2v = -torch.sum(F.log_softmax(sim_v2v, dim=1)*sim_targets,dim=1).mean()
                    # loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets,dim=1).mean()

                    vtc_loss = (loss_v2t + loss_t2v) / 2

                    loss = pred_loss + vtc_loss

                    loss.backward()
                    self.optimizer.step()

                    detached_loss = loss.detach().item()

                    if self.writer is not None:
                        self.writer.log_scalar(f'loss_train_{dl_idx}', detached_loss)

                    total_loss[dl_idx] += detached_loss

                    progress.set_postfix({"dl": dl_idx, "loss": detached_loss})

                    self.optimizer.zero_grad()

                if batch_idx == self.len_epoch:
                    break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log, results, truths = self._valid_epoch(bTest=False)
            log.update(val_log)

            dataset_name = self.config["data_loader"]["args"]["dataset_name"]
            if dataset_name == "SIMS" or dataset_name == "SIMS_V2":
                val_acc = eval_sims_regression(results, truths)
            else:
                val_acc = eval_mosi(results, truths)

            val_acc_log = {'val_acc2': val_acc}
            log.update(val_acc_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, bTest=False):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """

        self.model.eval()
        if bTest == False:
            total_loss = [0] * len(self.valid_data_loader)
            data_loader = self.valid_data_loader
        else:
            total_loss = [0] * len(self.test_data_loader)
            data_loader = self.test_data_loader
        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(data_loader):
                results = []
                truths = []

                statu = "Testing" if bTest else "Validating"
                for data in tqdm(dl, desc=f"{statu} dl{dl_idx}"):
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)
                    data['video_aug'] = data['video_aug'].to(self.device)
                    data['label'] = data['label'].to(self.device)

                    # Note that if the batch is not scattered among all the GPUs, `DataParallel` will fail because
                    # the model's mandatory argument `data` will not be passed to some of them.
                    # It can happen with the last batch of the dataset, depending on its size.
                    # It could be safely ignored during training but on validation/test we want accurate metrics.
                    # This avoids using `DataParallel` in this case, and supposes this batch fits in one GPU.
                    current_batch_size = data['video'].shape[0]
                    if isinstance(self.model, nn.DataParallel) and current_batch_size < (dl.batch_size or 1):
                        scattered_len = len(self.model.scatter([torch.empty(current_batch_size)], {},
                                                               self.model.device_ids)[0])
                        avoid_data_parallel = scattered_len < len(self.model.device_ids)
                    else:
                        avoid_data_parallel = False

                    if avoid_data_parallel:
                        preds = self.model.module(data)
                    else:
                        preds = self.model(data)

                    loss = self.criterion(preds, data['label'])

                    total_loss[dl_idx] += loss.item()

                    # Collect the results into dictionary
                    results.append(preds)
                    truths.append(data['label'])

        res_dict = {f'val_loss_{dl_idx}': total_loss[dl_idx] / len(data_loader[dl_idx])
                    for dl_idx in range(len(data_loader))}

        results = torch.cat(results)
        truths = torch.cat(truths)

        return res_dict, results, truths

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
