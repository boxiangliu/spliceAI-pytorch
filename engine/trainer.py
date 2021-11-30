import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import H5Dataset
from utils import load_config, tensor2numpy, get_topl_statistics
from model.model import SpliceAI, L, W, AR
import time
import logging
import shutil
import sys
import os
import wandb


class Trainer(object):

    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.initialize()

    def initialize(self):
        self.cfg = load_config(self.cfg_file)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        sys.stderr.write("Loading data...\n")
        self.train_data = H5Dataset(self.cfg.DATA.TRAIN)
        self.dev_data = H5Dataset(self.cfg.DATA.DEV)

        if self.cfg.DEBUG:
            self.dev_data.num_examples = 100

        self.train_loader = DataLoader(
            dataset=self.train_data,
            shuffle=True,
            batch_size=self.cfg.PARAMS.LOADER.BATCH,
            num_workers=self.cfg.PARAMS.LOADER.WORKERS)
        self.train_iter = iter(self.train_loader)

        self.dev_loader = DataLoader(
            dataset=self.dev_data,
            shuffle=False,
            batch_size=self.cfg.PARAMS.LOADER.BATCH,
            num_workers=self.cfg.PARAMS.LOADER.WORKERS)

        if self.cfg.MODEL == "spliceAI":
            self.model = SpliceAI(L, W, AR).to(self.device)
        else:
            raise ValueError(f"{self.cfg.MODEL} not implemented!")

        if self.cfg.PARAMS.OPTIMIZER.NAME == "ADAM":
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.cfg.PARAMS.OPTIMIZER.LR,
                weight_decay=self.cfg.PARAMS.OPTIMIZER.WEIGHT_DECAY)
        else:
            raise ValueError(f"{self.cfg.PARAMS.OPTIMIZER.NAME} not implemented!")

        if self.cfg.PARAMS.LOSS == "CE":
            self.loss_fun = nn.CrossEntropyLoss(reduction="none")

        self.summary = {
            "epoch": 0,
            "step": 0,
            "log_step": 0,
            "train_loss_sum": 0.0,
            "train_topl_acc_1_sum": 0.0,
            "train_threshold_1_sum": 0.0,
            "train_auprc_1_sum": 0.0,
            "train_pos_label_1_sum": 0.0,
            "train_topl_acc_2_sum": 0.0,
            "train_threshold_2_sum": 0.0,
            "train_auprc_2_sum": 0.0,
            "train_pos_label_2_sum": 0.0,
            "dev_loss": 0.0,
            "dev_loss_best": float("inf"),
            "dev_topl_acc_1": 0.0,
            "dev_threshold_1": 0.0,
            "dev_auprc_1": 0.0,
            "dev_pos_label_1": 0.0,
            "dev_topl_acc_2": 0.0,
            "dev_threshold_2": 0.0,
            "dev_auprc_2": 0.0,
            "dev_pos_label_2": 0.0
        }

        os.makedirs(self.cfg.LOGGING.DIR, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{self.cfg.LOGGING.DIR}/log.txt"),
                logging.StreamHandler()
            ])

        shutil.copyfile(self.cfg_file, f"{self.cfg.LOGGING.DIR}/config.yaml")

        wandb.init(
            project="spliceAI",
            name=self.cfg.METADATA.NAME,
            notes=self.cfg.METADATA.DESCRIPTION,
            config={
                "train_data": self.cfg.DATA.TRAIN,
                "dev_data": self.cfg.DATA.DEV,
                "model": self.cfg.MODEL,
                "loss": self.cfg.PARAMS.LOSS,
                "optimizer": self.cfg.PARAMS.OPTIMIZER.NAME,
                "lr": self.cfg.PARAMS.OPTIMIZER.LR,
                "weight_decay": self.cfg.PARAMS.OPTIMIZER.WEIGHT_DECAY,
                "momentum": self.cfg.PARAMS.OPTIMIZER.MOMENTUM,
                "batch_size": self.cfg.PARAMS.LOADER.BATCH,
                "epoch": self.cfg.PARAMS.EPOCH
            })

        self.time_stamp = time.time()

    def reset_summary(self):
        self.summary["train_loss_sum"] = 0.0
        self.summary["log_step"] = 0

        self.summary["train_topl_acc_1_sum"] = 0.0
        self.summary["train_threshold_1_sum"] = 0.0
        self.summary["train_auprc_1_sum"] = 0.0
        self.summary["train_pos_label_1_sum"] = 0.0

        self.summary["train_topl_acc_2_sum"] = 0.0
        self.summary["train_threshold_2_sum"] = 0.0
        self.summary["train_auprc_2_sum"] = 0.0
        self.summary["train_pos_label_2_sum"] = 0.0

    def stats_fun(self, outputs, labels):
        is_expr = (labels.sum(axis=(1, 2)) >= 1)

        Y_true_1 = labels[is_expr, 1, :].flatten()
        Y_true_2 = labels[is_expr, 2, :].flatten()
        Y_pred_1 = outputs[is_expr, 1, :].flatten()
        Y_pred_2 = outputs[is_expr, 2, :].flatten()

        stats_1 = get_topl_statistics(Y_true_1, Y_pred_1)
        stats_2 = get_topl_statistics(Y_true_2, Y_pred_2)

        return stats_1, stats_2

    def train_step(self):
        try:
            seqs, labels = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            seqs, labels = next(self.train_iter)
            self.summary["epoch"] += 1

        seqs = seqs.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(seqs)
        loss = self.loss_fun(outputs, labels).mean()
        stats = self.stats_fun(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.summary["train_loss_sum"] += tensor2numpy(loss)

        for i, stats_i in enumerate(stats):
            topl_accuracy, threshold, auprc, pos_label = stats_i
            self.summary[f"train_topl_acc_{i+1}_sum"] += topl_accuracy[1]
            self.summary[f"train_threshold_{i+1}_sum"] += threshold[1]
            self.summary[f"train_auprc_{i+1}_sum"] += auprc
            self.summary[f"train_pos_label_{i+1}_sum"] += pos_label

        self.summary["step"] += 1
        self.summary["log_step"] += 1

    def dev_epoch(self):
        self.model.eval()

        dev_summary = {
            "dev_loss_sum": 0.0,
            "dev_topl_acc_1_sum": 0.0,
            "dev_threshold_1_sum": 0.0,
            "dev_auprc_1_sum": 0.0,
            "dev_pos_label_1_sum": 0.0,
            "dev_topl_acc_2_sum": 0.0,
            "dev_threshold_2_sum": 0.0,
            "dev_auprc_2_sum": 0.0,
            "dev_pos_label_2_sum": 0.0
        }

        with torch.no_grad():
            for seqs, labels in self.dev_loader:
                seqs = seqs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(seqs)
                loss = self.loss_fun(outputs, labels).mean(axis=1)
                stats = self.stats_fun(outputs, labels)

                dev_summary["dev_loss_sum"] += tensor2numpy(loss.sum())

                for i, stats_i in enumerate(stats):
                    topl_accuracy, threshold, auprc, pos_label = stats[0]
                    dev_summary[f"dev_topl_acc_{i+1}_sum"] += topl_accuracy[1]
                    dev_summary[f"dev_threshold_{i+1}_sum"] += threshold[1]
                    dev_summary[f"dev_auprc_{i+1}_sum"] += auprc
                    dev_summary[f"dev_pos_label_{i+1}_sum"] += pos_label

        self.summary["dev_loss"] = dev_summary["dev_loss_sum"] / len(self.dev_data)

        for i in [1, 2]:
            self.summary[f"dev_topl_acc_{i+1}"] = dev_summary[f"dev_topl_acc_{i+1}_sum"] / len(self.dev_loader)
            self.summary[f"dev_threshold_{i+1}"] = dev_summary[f"dev_threshold_{i+1}_sum"] / len(self.dev_loader)
            self.summary[f"dev_auprc_{i+1}"] = dev_summary[f"dev_auprc_{i+1}_sum"] / len(self.dev_loader)
            self.summary[f"dev_pos_label_{i+1}"] = dev_summary[f"dev_pos_label_{i+1}_sum"] / len(self.dev_loader)

        self.model.train()

    def log(self, mode="train"):
        elapsed_time = time.time() - self.time_stamp
        self.time_stamp = time.time()
        log_step = self.summary["log_step"]

        if mode == "train":
            train_loss = self.summary["train_loss_sum"] / log_step

            train_topl_acc_1 = self.summary["train_topl_acc_1_sum"] / log_step
            train_threshold_1 = self.summary["train_threshold_1_sum"] / log_step
            train_auprc_1 = self.summary["train_auprc_1_sum"] / log_step
            train_pos_label_1 = self.summary["train_pos_label_1_sum"] / log_step

            train_topl_acc_2 = self.summary["train_topl_acc_2_sum"] / log_step
            train_threshold_2 = self.summary["train_threshold_2_sum"] / log_step
            train_auprc_2 = self.summary["train_auprc_2_sum"] / log_step
            train_pos_label_2 = self.summary["train_pos_label_2_sum"] / log_step

            logging.info("TRAIN, Epoch: {}, Step: {}, Loss: {}, "
                         "TopL_1: {}, Threshold_1: {}, AUPRC_1: {}, #Pos_1: {}, "
                         "TopL_2: {}, Threshold_2: {}, AUPRC_2: {}, #Pos_2: {}, "
                         "Time: {:.2f} s".format(
                             self.summary["epoch"], self.summary["step"], train_loss,
                             train_topl_acc_1, train_threshold_1, train_auprc_1, train_pos_label_1,
                             train_topl_acc_2, train_threshold_2, train_auprc_2, train_pos_label_2,
                             elapsed_time
                         ))

            wandb.log({"train": {"loss": train_loss,
                                 "topL_1": train_topl_acc_1,
                                 "threshold_1": train_threshold_1,
                                 "AUPRC_1": train_auprc_1,
                                 "#pos_1": train_pos_label_1,
                                 "topL_2": train_topl_acc_2,
                                 "threshold_2": train_threshold_2,
                                 "AUPRC_2": train_auprc_2,
                                 "#pos_2": train_pos_label_2}},
                      step=self.summary["step"])

            self.reset_summary()

        elif mode == "dev":
            logging.info("DEV, EPOCH: {}, STEP: {}, LOSS: {}, "
                         "TopL_1: {}, Threshold_1: {}, AUPRC_1: {}, #Pos_1: {}, "
                         "TopL_2: {}, Threshold_2: {}, AUPRC_2: {}, #Pos_2: {}, "
                         "TIME: {:.2f} s".format(
                             self.summary["epoch"], self.summary["step"], self.summary["dev_loss"],
                             self.summary["dev_topl_acc_1"], self.summary["dev_threshold_1"],
                             self.summary["dev_auprc_1"], self.summary["dev_pos_label_1"],
                             self.summary["dev_topl_acc_2"], self.summary["dev_threshold_2"],
                             self.summary["dev_auprc_2"], self.summary["dev_pos_label_2"],
                             elapsed_time
                         ))

            wandb.log({"dev": {"loss": self.summary["dev_loss"],
                               "topL_1": self.summary["dev_topl_acc_1"],
                               "threshold_1": self.summary["dev_threshold_1"],
                               "AUPRC_1": self.summary["dev_auprc_1"],
                               "#pos_1": self.summary["dev_pos_label_1"],
                               "topL_2": self.summary["dev_topl_acc_2"],
                               "threshold_2": self.summary["dev_threshold_2"],
                               "AUPRC_2": self.summary["dev_auprc_2"],
                               "#pos_2": self.summary["dev_pos_label_2"]}},
                      step=self.summary["step"])

    def save(self, mode="train"):
        if mode == "train":
            torch.save({
                "epoch": self.summary["epoch"],
                "step": self.summary["step"],
                "state_dict": self.model.state_dict()
            }, os.path.join(self.cfg.LOGGING.DIR, 'train.ckpt'))

        elif mode == "dev":
            if self.summary["dev_loss"] < self.summary["dev_loss_best"]:
                self.summary["dev_loss_best"] = self.summary["dev_loss"]
                torch.save({
                    "epoch": self.summary["epoch"],
                    "step": self.summary["step"],
                    "dev_loss_best": self.summary["dev_loss_best"],
                    "state_dict": self.model.state_dict()
                }, os.path.join(self.cfg.LOGGING.DIR, 'best.ckpt'))

                logging.info("BEST, EPOCH: {}, STEP: {}, LOSS: {}".format(
                    self.summary["epoch"],
                    self.summary["step"],
                    self.summary["dev_loss_best"]
                ))

    def finish(self):
        wandb.finish()
