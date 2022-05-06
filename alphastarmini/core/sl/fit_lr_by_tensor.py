#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train from the replay files through python tensor (pt) file"

import argparse
import datetime
import gc
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from absl import app, flags
from tensorboardX import SummaryWriter
from torch.optim import AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import param as P
from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.sl import sl_loss_multi_gpu as Loss
from alphastarmini.core.sl import sl_utils as SU
from alphastarmini.core.sl.dataset import ReplayTensorDataset
from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import \
    SL_Training_Hyper_Parameters as SLTHP
from alphastarmini.lib.hyper_parameters import \
    StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.utils import initial_model_state_dict, load_latest_model

__author__ = "Ruo-Ze Liu"

debug = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p1",
    "--path1",
    default="/home/cozy/Documents/projects/sc2_rl/mini-AlphaStar/data/replay_data_tensor_new_small/",
    help="The path where data stored",
)
parser.add_argument(
    "-p2",
    "--path2",
    default="./data/replay_data_tensor_new_small_AR/",
    help="The path where data stored",
)
parser.add_argument(
    "-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type"
)
parser.add_argument(
    "-r",
    "--restore",
    action="store_true",
    default=False,
    help="whether to restore model or not",
)
parser.add_argument(
    "-c", "--clip", action="store_true", default=False, help="whether to use clipping"
)
parser.add_argument("--num_workers", type=int, default=2, help="")


args = parser.parse_args()

# training paramerters
if SCHP.map_name == "Simple64":
    PATH = args.path1
elif SCHP.map_name == "AbyssalReef":
    PATH = args.path2
else:
    raise Exception

MODEL = args.model
RESTORE = args.restore
CLIP = args.clip
NUM_WORKERS = args.num_workers

MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

MODEL_PATH_TRAIN = "./model_train/"
if not os.path.exists(MODEL_PATH_TRAIN):
    os.mkdir(MODEL_PATH_TRAIN)

RESTORE_NAME = "sl_21-12-21_09-11-12"
RESTORE_PATH = MODEL_PATH + RESTORE_NAME + ".pth"
RESTORE_PATH_TRAIN = MODEL_PATH_TRAIN + RESTORE_NAME + ".pkl"

SAVE_STATE_DICT = True
SAVE_ALL_PKL = False
SAVE_CHECKPOINT = True

LOAD_STATE_DICT = False
LOAD_ALL_PKL = False
LOAD_CHECKPOINT = True

TRAIN_FROM = 0  # 20
TRAIN_NUM = 90  # 60

VAL_FROM = 0
VAL_NUM = 5

# hyper paramerters
# use the same as in RL
# BATCH_SIZE = AHP.batch_size
# SEQ_LEN = AHP.sequence_length

# important: use larger batch_size and smaller seq_len in SL!
BATCH_SIZE = 3 * AHP.batch_size
SEQ_LEN = int(AHP.sequence_length * 0.5)

print('BATCH_SIZE:', BATCH_SIZE) if debug else None
print('SEQ_LEN:', SEQ_LEN) if debug else None

NUM_EPOCHS = 100
LEARNING_RATE = 1e-8
WEIGHT_DECAY = 1e-5

CLIP_VALUE = 0.5  # SLTHP.clip
STEP_SIZE = 30
GAMMA = 0.2

torch.manual_seed(SLTHP.seed)
np.random.seed(SLTHP.seed)


def getReplayData(path, replay_files, from_index=0, end_index=None):
    td_list = []
    for i, replay_file in enumerate(tqdm(replay_files)):
        try:
            replay_path = path + replay_file
            # print('replay_path:', replay_path) if 1 else None

            do_write = False
            if i >= from_index:
                if end_index is None:
                    do_write = True
                elif end_index is not None and i < end_index:
                    do_write = True

            if not do_write:
                continue

            features, labels = torch.load(replay_path)
            print("features.shape:", features.shape) if debug else None
            print("labels.shape::", labels.shape) if debug else None

            replay_ds = ReplayTensorDataset(features, labels, seq_len=SEQ_LEN)
            # why does it only work with this???
            # print(len(replay_ds))
            td_list.append(replay_ds)

        except Exception as e:
            traceback.print_exc()

    return td_list


def main_worker(device):
    print("==> Making model..")
    net = ArchModel()
    checkpoint = None
    if RESTORE:
        if LOAD_STATE_DICT:
            # use state dict to restore
            net.load_state_dict(
                torch.load(RESTORE_PATH, map_location=device), strict=False
            )

        if LOAD_ALL_PKL:
            # use all to restore
            net = torch.load(RESTORE_PATH_TRAIN, map_location=device)

        if LOAD_CHECKPOINT:
            # use checkpoint to restore
            checkpoint = torch.load(RESTORE_PATH_TRAIN, map_location=device)
            net.load_state_dict(checkpoint["model"], strict=False)

    net = net.to(device)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("The number of parameters of model is", num_params)

    print("==> Making optimizer and scheduler..")

    optimizer, scheduler = None, None
    batch_iter, epoch = 0, 0

    if RESTORE and LOAD_CHECKPOINT:
        # use checkpoint to restore other
        optimizer = AdamW(net.parameters())
        optimizer.load_state_dict(checkpoint["optimizer"])

        # scheduler = StepLR(optimizer, step_size=STEP_SIZE)
        scheduler.load_state_dict(checkpoint["scheduler"])

        batch_iter = checkpoint["batch_iter"]
        print("batch_iter is", batch_iter)
        epoch = checkpoint["epoch"]
        print("epoch is", epoch)

        ckpt = torch.load(RESTORE_PATH_TRAIN)
        np.random.set_state(ckpt["numpy_random_state"])
        torch.random.set_rng_state(ckpt["torch_random_state"])
    else:
        optimizer = AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("==> Preparing data..")

    replay_files = os.listdir(PATH)
    print("length of replay_files:", len(replay_files)) if debug else None
    replay_files.sort()

    train_list = getReplayData(
        PATH, replay_files, from_index=TRAIN_FROM, end_index=TRAIN_FROM + TRAIN_NUM
    )
    val_list = getReplayData(
        PATH, replay_files, from_index=VAL_FROM, end_index=VAL_FROM + VAL_NUM
    )

    print("len(train_list)", len(train_list)) if debug else None
    print("len(val_list)", len(val_list)) if debug else None

    train_set = ConcatDataset(train_list)
    val_set = ConcatDataset(val_list)

    print("len(train_set)", len(train_set)) if debug else None
    print("len(val_set)", len(val_set)) if debug else None

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print("len(train_loader)", len(train_loader)) if debug else None
    print("len(val_loader)", len(val_loader)) if debug else None

    steps_per_epoch = 0
    for _ in enumerate(train_loader):
        steps_per_epoch += 1

    # scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=NUM_EPOCHS * steps_per_epoch)

    train(
        net,
        optimizer,
        scheduler,
        train_set,
        train_loader,
        device,
        val_set,
        batch_iter,
        epoch,
        val_loader,
    )


def train(
    net,
    optimizer,
    scheduler,
    train_set,
    train_loader,
    device,
    val_set,
    batch_iter,
    epoch,
    val_loader=None,
):

    now = datetime.datetime.now()
    summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    writer = SummaryWriter(summary_path)

    time_str = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
    SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time_str)
    SAVE_PATH_TRAIN = os.path.join(MODEL_PATH_TRAIN, MODEL + "_" + time_str)

    epoch_start = time.time()

    losses = []
    lrs = []
    loss_sum = 0
    epoch += 1

    # put model in train mode
    net.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        while True:
            start = time.time()

            feature_tensor = features.to(device).float()
            labels_tensor = labels.to(device).float()

            loss, loss_list, acc_num_list = Loss.get_sl_loss_for_tensor(
                feature_tensor,
                labels_tensor,
                net,
                decrease_smart_opertaion=True,
                return_important=True,
                only_consider_small=False,
                train=True,
            )
            del feature_tensor, labels_tensor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for g in optimizer.param_groups:
                print(g["lr"])
                g["lr"] = g["lr"] * 5
                if float(loss.item()) >= 1000 or g["lr"] >= 1.:
                    import matplotlib.pyplot as plt
                    lrs = [str(round(lr, 6)) for lr in lrs]
                    print(lrs)
                    print(losses)
                    plt.xticks(rotation=45)
                    plt.plot(lrs, losses)
                    plt.savefig("fit_lr.png")
                    plt.show()
                    sys.exit()
                else:
                    losses.append(float(loss.item()))
                    lrs.append(g["lr"])

            # add a grad clip
            if CLIP:
                parameters = [
                    p for p in net.parameters() if p is not None and p.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(parameters, CLIP_VALUE)

            loss_value = float(loss.item())

            loss_sum += loss_value
            del loss

            action_accuracy = acc_num_list[0] / (acc_num_list[1] + 1e-9)
            move_camera_accuracy = acc_num_list[2] / (acc_num_list[3] + 1e-9)
            non_camera_accuracy = acc_num_list[4] / (acc_num_list[5] + 1e-9)
            short_important_accuracy = acc_num_list[6] / (acc_num_list[7] + 1e-9)

            location_accuracy = acc_num_list[8] / (acc_num_list[9] + 1e-9)
            location_distance = acc_num_list[11] / (acc_num_list[9] + 1e-9)

            selected_units_accuracy = acc_num_list[12] / (acc_num_list[13] + 1e-9)
            selected_units_type_right = acc_num_list[14] / (acc_num_list[15] + 1e-9)
            selected_units_num_right = acc_num_list[16] / (acc_num_list[17] + 1e-9)

            target_unit_accuracy = acc_num_list[18] / (acc_num_list[19] + 1e-9)

            batch_time = time.time() - start

            batch_iter += 1
            print("batch_iter", batch_iter)

            gc.collect()

            print(
                "Batch/Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s ".format(
                    batch_iter, epoch, loss_value, action_accuracy, batch_time
                )
            )


def test(on_server):
    # gpu setting
    ON_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")

    if ON_GPU:
        if torch.backends.cudnn.is_available():
            print("cudnn available")
            print("cudnn version", torch.backends.cudnn.version())
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    main_worker(DEVICE)
