import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import deque
import random
import math

from deep_rl.trainer import Transition as Transition
from deep_rl.deep_q_agent import DeepQAgent


class DoubleDeepQAgent(DeepQAgent):
    def __init__(self, network_generator, optim_class, nb_actions: int):
        super().__init__(network_generator, optim_class, nb_actions)
        self.optimizer = optim_class(self.policy_net.parameters())

    def update_epsilon(self, episode_nb):
        diff = DoubleDeepQAgent.EPSILON_START - DoubleDeepQAgent.EPSILON_END
        self.epsilon = DoubleDeepQAgent.EPSILON_END + diff * \
            math.exp(-1 * episode_nb / DoubleDeepQAgent.EPSILON_DECAY)

    def observe_transition(self, transition, device):
        self.replay_buffer.append(transition)

        if len(self.replay_buffer) <= self.batch_size * 10:
            return

        transitions = random.sample(self.replay_buffer, self.batch_size)
        mini_batch = Transition(*zip(*transitions))

        state_batch = torch.stack(mini_batch.state)
        action_batch = torch.stack(mini_batch.action)
        action_batch = action_batch.unsqueeze(1)
        reward_batch = torch.stack(mini_batch.reward)
        next_state_batch = torch.stack(mini_batch.next_state)
        non_final_mask = ~torch.stack(mini_batch.done)

        # 两步
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_actions = torch.argmax(
            self.policy_net(next_state_batch), dim=1)
        next_state_values[non_final_mask] = self.target_net(next_state_batch)[
            torch.arange(self.batch_size), next_state_actions
        ][non_final_mask]
        expected_values = (next_state_values *
                           DeepQAgent.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values,
                                expected_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_counter % DeepQAgent.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1
