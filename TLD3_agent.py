from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append('../../')
from TLD3_networks import ActorNet, CriticNet


class Agent:

    def __init__(self,
                 state_num,
                 action_num,
                 rescale_state_num,
                 actor_net_dim=(256, 256, 256,128,1),
                 critic_net_dim=(512, 512, 512),
                 memory_size=1000,
                 batch_size=128,
                 target_tau=0.01,
                 target_update_steps=5,
                 reward_gamma=0.99,
                 actor_lr=0.0001,
                 critic_lr=0.0001,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 epsilon_rand_decay_start=60000,
                 epsilon_rand_decay_step=1,
                 poisson_window=50,
                 use_poisson=False,
                 use_cuda=True):

        self.state_num = state_num 
        self.action_num = action_num 
        self.rescale_state_num = rescale_state_num 
        self.memory_size = memory_size 
        self.batch_size = batch_size 
        self.target_tau = target_tau 
        self.target_update_steps = target_update_steps 
        self.reward_gamma = reward_gamma 
        self.actor_lr = actor_lr 
        self.critic_lr = critic_lr 
        self.epsilon_start = epsilon_start 
        self.epsilon_end = epsilon_end 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_rand_decay_start = epsilon_rand_decay_start 
        self.epsilon_rand_decay_step = epsilon_rand_decay_step 
        self.poisson_window = poisson_window 
        self.use_poisson = use_poisson 
        self.use_cuda = use_cuda 

        self.epsilon = epsilon_start 

        if self.use_cuda: 
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.memory = deque(maxlen=self.memory_size) 

        self.actor_net = ActorNet(self.rescale_state_num, self.action_num,
                                  hidden1=actor_net_dim[0],
                                  hidden2=actor_net_dim[1],
                                  hidden3=actor_net_dim[2],
                                  lstm_hidden =actor_net_dim[3],
                                  num_layers =actor_net_dim[4])
        self.critic_net = CriticNet(self.state_num, self.action_num,
                                    hidden1=critic_net_dim[0],
                                    hidden2=critic_net_dim[1],
                                    hidden3=critic_net_dim[2],
                                    )
        self.target_actor_net = ActorNet(self.rescale_state_num, self.action_num,
                                         hidden1=actor_net_dim[0],
                                         hidden2=actor_net_dim[1],
                                         hidden3=actor_net_dim[2],
                                         lstm_hidden =actor_net_dim[3],
                                         num_layers =actor_net_dim[4])
        self.target_critic_net = CriticNet(self.state_num, self.action_num,
                                           hidden1=critic_net_dim[0],
                                           hidden2=critic_net_dim[1],
                                           hidden3=critic_net_dim[2],
                                           )
        self._hard_update(self.target_actor_net, self.actor_net)
        self._hard_update(self.target_critic_net, self.critic_net) 
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.target_actor_net.to(self.device)
        self.target_critic_net.to(self.device) 

        self.criterion = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr) 

        self.step_ita = 0 

    def remember(self, state, rescale_state, action, reward, next_state, rescale_next_state, done):

        self.memory.append((state, rescale_state, action, reward, next_state, rescale_next_state, done))

    def act(self, state, explore=True, train=True):

        with torch.no_grad():
            state = np.array(state)
            if self.use_poisson:
                state = self._state_2_poisson_state(state, 1)
            state = torch.Tensor(state.reshape((1, -1))).to(self.device)
            action = self.actor_net(state).to('cpu')
            action = action.numpy().squeeze()
        if train:
            if self.step_ita > self.epsilon_rand_decay_start and self.epsilon > self.epsilon_end:
                if self.step_ita % self.epsilon_rand_decay_step == 0:
                    self.epsilon = self.epsilon * self.epsilon_decay
            noise = np.random.randn(self.action_num) * self.epsilon
            action = noise + (1 - self.epsilon) * action
            action = np.clip(action, [0., 0.], [1., 1.])
        elif explore:
            noise = np.random.randn(self.action_num) * self.epsilon_end
            action = noise + (1 - self.epsilon_end) * action
            action = np.clip(action, [0., 0.], [1., 1.])
        return action.tolist()

    def replay(self):

        state_batch, r_state_batch, action_batch, reward_batch, nstate_batch, r_nstate_batch, done_batch = self._random_minibatch()

        with torch.no_grad():
            naction_batch = self.target_actor_net(r_nstate_batch)
            next_q1,next_q2 = self.target_critic_net([nstate_batch, naction_batch])
            next_q = torch.min(next_q1,next_q2)
            target_q = reward_batch + self.reward_gamma * next_q * (1. - done_batch)

        self.critic_optimizer.zero_grad()
        current_q_1,current_q_2 = self.critic_net([state_batch, action_batch])
        critic_loss_1 = self.criterion(current_q_1, target_q)
        critic_loss_2 = self.criterion(current_q_2, target_q)
        critic_loss = critic_loss_1 + critic_loss_2
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        current_action = self.actor_net(r_state_batch)
        actor_loss , _ =  self.critic_net([state_batch, current_action])
        actor_loss = -actor_loss.mean()
        actor_loss_item = actor_loss.item()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.step_ita += 1
        if self.step_ita % self.target_update_steps == 0:
            self._soft_update(self.target_actor_net, self.actor_net)
            self._soft_update(self.target_critic_net, self.critic_net)
        return actor_loss_item, critic_loss_item

    def reset_epsilon(self, new_epsilon, new_decay):

        self.epsilon = new_epsilon
        self.epsilon_decay = new_decay

    def save(self, save_dir, episode, run_name):

        try:
            os.mkdir(save_dir)
            print("Directory ", save_dir, " Created")
        except FileExistsError:
            print("Directory", save_dir, " already exists")
        torch.save(self.actor_net.state_dict(),
                   save_dir + '/' + run_name + '_actor_network_s' + str(episode) + '.pt')
        print("Episode " + str(episode) + " weights saved ...")

    def load(self, load_file_name):

        self.actor_net.to('cpu')
        self.actor_net.load_state_dict(torch.load(load_file_name))
        self.actor_net.to(self.device)

    def _state_2_poisson_state(self, state_value, batch_size):

        spike_state_value = state_value.reshape((batch_size, self.rescale_state_num, 1))
        state_spikes = np.random.rand(batch_size, self.rescale_state_num, self.poisson_window) < spike_state_value
        poisson_state = np.sum(state_spikes, axis=2).reshape((batch_size, -1))
        poisson_state = poisson_state / self.poisson_window
        poisson_state = poisson_state.astype(float)
        return poisson_state

    def _random_minibatch(self):

        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = np.zeros((self.batch_size, self.state_num))
        rescale_state_batch = np.zeros((self.batch_size, self.rescale_state_num))
        action_batch = np.zeros((self.batch_size, self.action_num))
        reward_batch = np.zeros((self.batch_size, 1))
        nstate_batch = np.zeros((self.batch_size, self.state_num))
        rescale_nstate_batch = np.zeros((self.batch_size, self.rescale_state_num))
        done_batch = np.zeros((self.batch_size, 1))
        for num in range(self.batch_size):
            state_batch[num, :] = np.array(minibatch[num][0])
            rescale_state_batch[num, :] = np.array(minibatch[num][1])
            action_batch[num, :] = np.array(minibatch[num][2])
            reward_batch[num, 0] = minibatch[num][3]
            nstate_batch[num, :] = np.array(minibatch[num][4])
            rescale_nstate_batch[num, :] = np.array(minibatch[num][5])
            done_batch[num, 0] = minibatch[num][6]
        if self.use_poisson:
            rescale_state_batch = self._state_2_poisson_state(rescale_state_batch, self.batch_size)
            rescale_nstate_batch = self._state_2_poisson_state(rescale_nstate_batch, self.batch_size)
        state_batch = torch.Tensor(state_batch).to(self.device)
        rescale_state_batch = torch.Tensor(rescale_state_batch).to(self.device)
        action_batch = torch.Tensor(action_batch).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device)
        nstate_batch = torch.Tensor(nstate_batch).to(self.device)
        rescale_nstate_batch = torch.Tensor(rescale_nstate_batch).to(self.device)
        done_batch = torch.Tensor(done_batch).to(self.device)
        return state_batch, rescale_state_batch, action_batch, reward_batch, nstate_batch, rescale_nstate_batch, done_batch

    def _hard_update(self, target, source):

        with torch.no_grad(): 
            for target_param, param in zip(target.parameters(), source.parameters()): 
                target_param.data.copy_(param.data) 

    def _soft_update(self, target, source):

        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau
                )
