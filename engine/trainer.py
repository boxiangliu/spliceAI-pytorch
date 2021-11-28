import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import H5Dataset
from utils import load_config, tensor2numpy
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

        self.acc_fun = lambda x, y: x.argmax(dim=1) == y.argmax(dim=1)

        self.summary = {
            "epoch": 0,
            "step": 0,
            "log_step": 0,
            "train_loss_sum": 0.0,
            "dev_loss": 0.0,
            "dev_loss_best": float("inf")
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
        self.summary["train_loss_sum"] = 0.0
        self.summary["log_step"] = 0

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
        breakpoint()
        acc = self.acc_fun(outputs, labels).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.summary["train_loss_sum"] += tensor2numpy(loss)
        self.summary["train_acc_sum"] += tensor2numpy(acc)
        self.summary["step"] += 1
        self.summary["log_step"] += 1

    def dev_epoch(self):
        self.model.eval()
        dev_loss_sum = 0.0
        dev_acc_sum = 0.0

        with torch.no_grad():
            for seqs, labels in self.dev_loader:
                seqs = seqs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(seqs)
                loss = self.loss_fun(outputs, labels).mean(axis=1)
                acc = self.acc_fun(outputs, labels).mean(axis=1)
                dev_loss_sum += tensor2numpy(loss.sum())

        self.summary["dev_loss"] = dev_loss_sum / len(self.dev_data)
        self.summary["dev_acc"] = dev_acc_sum / len(self.dev_data)
        self.model.train()

    def log(self, mode="train"):
        elapsed_time = time.time() - self.time_stamp
        self.time_stamp = time.time()

        if mode == "train":
            train_loss = \
                self.summary["train_loss_sum"] / self.summary["log_step"]
            train_acc = \
                self.summary["train_acc_sum"] / self.summary["log_step"]

            logging.info("TRAIN, EPOCH: {}, STEP: {}, LOSS: {}, ACC: {}, TIME: {:.2f} s".format(
                self.summary["epoch"],
                self.summary["step"],
                train_loss,
                train_acc,
                elapsed_time
            ))

            wandb.log({"train": {"loss": train_loss, "acc": train_acc}},
                      step=self.summary["step"])

            self.reset_summary()

        elif mode == "dev":
            logging.info("DEV, EPOCH: {}, STEP: {}, LOSS: {}, ACC: {}, TIME: {:.2f} s".format(
                self.summary["epoch"],
                self.summary["step"],
                self.summary["dev_loss"],
                self.summary["dev_acc"],
                elapsed_time
            ))

            wandb.log({"dev": {"loss": train_loss, "acc": train_acc}},
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
