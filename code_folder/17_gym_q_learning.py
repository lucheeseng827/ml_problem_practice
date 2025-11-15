"""
Q-Learning with OpenAI Gym
===========================
Category 17: Reinforcement Learning - Tabular Q-learning

Use cases: Game AI, robotics, control systems
"""

import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error


def simple_gridworld_env():
    """Simple gridworld: 4x4 grid, goal at (3,3)"""
    class GridWorld:
        def __init__(self):
            self.state = 0
            self.goal = 15
        
        def reset(self):
            self.state = 0
            return self.state
        
        def step(self, action):
            # 0:up, 1:right, 2:down, 3:left
            row, col = self.state // 4, self.state % 4
            
            if action == 0 and row > 0: row -= 1
            elif action == 1 and col < 3: col += 1
            elif action == 2 and row < 3: row += 1
            elif action == 3 and col > 0: col -= 1
            
            self.state = row * 4 + col
            reward = 1 if self.state == self.goal else -0.01
            done = self.state == self.goal
            
            return self.state, reward, done
    
    return GridWorld()


def main():
    print("=" * 60)
    print("Q-Learning with OpenAI Gym")
    print("=" * 60)
    
    env = simple_gridworld_env()
    agent = QLearningAgent(n_states=16, n_actions=4)
    
    print("\nTraining Q-learning agent...")
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        
        for step in range(50):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if (episode + 1) % 200 == 0:
            print(f"Episode {episode + 1}, Reward: {total_reward:.2f}")
    
    print("\nLearned Q-Table (first 4 states):")
    print(agent.q_table[:4])
    
    print("\nKey Takeaways:")
    print("- Q-learning learns optimal action-value function")
    print("- Exploration vs exploitation (epsilon-greedy)")
    print("- Temporal difference learning")
    print("- Model-free reinforcement learning")


if __name__ == "__main__":
    main()
