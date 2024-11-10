import numpy as np

class ExperienceReplay():
    def __init__(self, max_memory):
        self.memory = list()
        self.maxMemory = max_memory

    def __len__(self):
        return len(self.memory)

    def sample(self, sample_size):
        indices = np.random.randint(0, sample_size, size=sample_size)
        return [self.memory[idx] for idx in indices]


    def remember(self, transition, gameOver):
        self.memory.append([transition, gameOver])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]

    