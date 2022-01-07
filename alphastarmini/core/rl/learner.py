#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the learner in the actor-learner mode in the IMPALA architecture"

# modified from AlphaStar pseudo-code

from time import time, sleep, strftime, localtime

import gc
import os
import traceback
import threading
import itertools
import datetime
import random
import copy

import torch
from torch.optim import Adam, RMSprop

from tensorboardX import SummaryWriter

from alphastarmini.core.rl.rl_loss import loss_function

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import RL_Training_Hyper_Parameters as THP

__author__ = "Ruo-Ze Liu"

debug = False

# model path
MODEL = "rl"
MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + strftime("%y-%m-%d_%H-%M-%S", localtime()))


class Learner:
    """Learner worker that updates agent parameters based on trajectories."""

    def __init__(self, player, max_time_for_training=60 * 3, lr=THP.learning_rate,
                 is_training=True, buffer_lock=None, writer=None,
                 use_opponent_state=True, no_replay_learn=False, 
                 num_epochs=THP.num_epochs, count_of_batches=1,
                 buffer_size=10, use_random_sample=False):
        self.player = player
        self.player.set_learner(self)
        self.trajectories = []

        # PyTorch code
        self.optimizer = Adam(self.get_parameters(), 
                              lr=lr, betas=(THP.beta1, THP.beta2), 
                              eps=THP.epsilon, weight_decay=THP.weight_decay)

        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True                            # Daemonize thread

        self.max_time_for_training = max_time_for_training
        self.is_running = False

        self.is_rl_training = is_training

        self.buffer_lock = buffer_lock
        self.writer = writer

        self.use_opponent_state = use_opponent_state
        self.no_replay_learn = no_replay_learn

        self.num_epochs = num_epochs
        self.count_of_batches = count_of_batches
        self.buffer_size = buffer_size
        self.use_random_sample = use_random_sample

    def get_parameters(self):
        return self.player.agent.get_parameters()

    def send_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def update_parameters(self):
        agent = self.player.agent
        batch_size = AHP.batch_size
        sample_size = self.count_of_batches * batch_size

        for ep_id in range(self.num_epochs):
            # TODO: use random samping
            with self.buffer_lock:
                if self.use_random_sample:
                    random.shuffle(self.trajectories)
                trajectories = self.trajectories[:sample_size]

            if self.is_rl_training:
                agent.agent_nn.model.train()  # for BN and dropout
                print("begin backward") if debug else None

                for batch_id in range(self.count_of_batches):
                    print('len(trajectories)', len(trajectories))

                    update_trajectories = trajectories[batch_id * batch_size: (batch_id + 1) * batch_size]
                    print('len(update_trajectories)', len(update_trajectories))

                    # with torch.autograd.set_detect_anomaly(True):
                    loss = loss_function(agent, update_trajectories, self.use_opponent_state, self.no_replay_learn)
                    print("loss:", loss) if debug else None

                    self.optimizer.zero_grad()
                    loss.backward() 
                    self.optimizer.step()

                    self.writer.add_scalar('learner/loss', loss.item(), agent.steps)
                    agent.steps += AHP.batch_size * AHP.sequence_length

                    del loss, update_trajectories

                del trajectories
                gc.collect()

                agent.agent_nn.model.eval()  # for BN and dropout 
                print("end backward") if debug else None

                # we use new ways to save
                if agent.steps % (1 * AHP.batch_size * AHP.sequence_length) == 0:
                    torch.save(agent.agent_nn.model.state_dict(), SAVE_PATH + "" + ".pth")

        with self.buffer_lock:
            self.trajectories = self.trajectories[sample_size:]

    def start(self):
        self.thread.start()

    # background
    def run(self):
        try:
            start_time = time()
            self.is_running = True

            while time() - start_time < self.max_time_for_training:
                try:
                    # if at least one actor is running, the learner would not stop
                    actor_is_running = False
                    if len(self.player.actors) == 0:
                        actor_is_running = True

                    for actor in self.player.actors:
                        if actor.is_start:
                            actor_is_running = actor_is_running | actor.is_running
                        else:
                            actor_is_running = actor_is_running | 1

                    if actor_is_running:
                        print('learner trajectories size:', len(self.trajectories)) if debug else None

                        if len(self.trajectories) >= self.buffer_size * AHP.batch_size:
                            print("learner begin to update parameters") if debug else None
                            self.update_parameters()
                            print("learner end updating parameters") if debug else None

                        sleep(1)
                    else:
                        print("Actor stops!") if debug else None

                        print("Learner also stops!") if debug else None
                        return

                except Exception as e:
                    print("Learner.run() Exception cause break, Detials of the Exception:", e) if debug else None
                    print(traceback.format_exc())
                    break

        except Exception as e:
            print("Learner.run() Exception cause return, Detials of the Exception:", e) if debug else None

        finally:
            self.is_running = False


def test(on_server):
    pass
