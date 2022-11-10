import numpy as np

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def store(self, obs, action, reward):
        self.buffer.append((obs, action, reward))

    def sample(self, batch_size):
        if len(batch_size) != self.capacity:
            raise Exception("Replay buffer must be full before sampling!")

        indices = np.random.choice(capacity, batch_size)
        self.buffer[indices]

        return indices