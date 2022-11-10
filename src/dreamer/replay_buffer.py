import numpy as np

class ReplayBuffer():
    def __init__(self, capacity):
        self._capacity = capacity
        self._buffer = []

    def store(self, obs, action, reward, next_obs):
        if len(self._buffer) >= self._capacity:
            self._buffer.pop(0)
        
        self._buffer.append((obs, action, reward, next_obs))

    def sample(self, batch_size):
        if len(self._buffer) < self._capacity:
            raise Exception("Replay buffer must be full before sampling!")

        indices = np.random.choice(self._capacity, batch_size)
        samples = [self._buffer[i] for i in indices]

        return samples

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._buffer)