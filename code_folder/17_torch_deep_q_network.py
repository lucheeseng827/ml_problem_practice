"""
Deep Q-Network (DQN) with PyTorch
==================================
Category 17: Deep RL for complex state spaces

Use cases: Atari games, robotics
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


def main():
    print("=" * 60)
    print("Deep Q-Network (DQN)")
    print("=" * 60)
    
    state_dim, action_dim = 4, 2
    dqn = DQN(state_dim, action_dim)
    target_dqn = DQN(state_dim, action_dim)
    target_dqn.load_state_dict(dqn.state_dict())
    
    optimizer = torch.optim.Adam(dqn.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer()
    
    print("\nTraining DQN...")
    for episode in range(100):
        state = np.random.randn(state_dim)
        
        for step in range(20):
            # Epsilon-greedy action
            if random.random() < 0.1:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = dqn(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            
            # Simulate step
            next_state = np.random.randn(state_dim)
            reward = random.random()
            done = random.random() < 0.1
            
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # Training
            if len(replay_buffer) > 64:
                batch = replay_buffer.sample(64)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones)
                
                # Compute Q-values
                q_values = dqn(states).gather(1, actions.unsqueeze(1))
                next_q_values = target_dqn(next_states).max(1)[0]
                targets = rewards + 0.99 * next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(q_values.squeeze(), targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        if (episode + 1) % 20 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
            print(f"Episode {episode + 1} completed")
    
    print("\nKey Takeaways:")
    print("- DQN combines Q-learning with deep neural networks")
    print("- Experience replay improves sample efficiency")
    print("- Target network stabilizes training")
    print("- Enabled superhuman Atari game playing")


if __name__ == "__main__":
    main()
