import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cart_pole_module
from math import pi

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.replay_buffer = deque(maxlen=1000000)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            if len(self.replay_buffer) < batch_size:
                continue
            
            batch = random.sample(self.replay_buffer, batch_size)
            state, next_state, action, reward, done = zip(*batch)

            state = torch.FloatTensor(np.array(state)).to(device)
            next_state = torch.FloatTensor(np.array(next_state)).to(device)
            action = torch.FloatTensor(np.array(action)).to(device)
            reward = torch.FloatTensor(np.array(reward)).to(device).unsqueeze(1)
            done = torch.FloatTensor(np.array(done)).to(device).unsqueeze(1)

            next_action = self.actor_target(next_state)
            noise = torch.FloatTensor(next_action).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = reward + (1 - done) * discount * torch.min(target_Q1, target_Q2)

            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)

            critic_loss = nn.MSELoss()(current_Q1, target_Q.detach()) + nn.MSELoss()(current_Q2, target_Q.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                actor_loss = -self.critic1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, next_state, action, reward, done))

if __name__ == "__main__":
    
    env = cart_pole_module.CartPole("cfg.yaml")
    state_dim = 4
    action_dim = 1
    max_action = 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = TD3(state_dim, action_dim, max_action)

    episode_rewards = []
    for episode in range(500):
        env.reset()
        state = np.array(env.get_state())
        episode_reward = 0
        for step in range(200):
            action = policy.select_action(state)
            next_states = env.step(action)
            next_state = next_states[-1]
            reward = 0
            if abs(next_state[1] - pi) < pi / 8:
                reward = 1
            done = env.is_done()
            next_state = np.array(next_state)
            policy.add_to_replay_buffer(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)
        policy.train(200)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    # Saving the model
    torch.save(policy.actor.state_dict(), "actor.pth")
    torch.save(policy.critic1.state_dict(), "critic1.pth")
    torch.save(policy.critic2.state_dict(), "critic2.pth")
