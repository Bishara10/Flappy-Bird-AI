import torch
import torch.nn as nn
import torch.optim as optim

class Brain(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Brain, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Example usage
input_size = 3
hidden_size = 20
output_size = 1
model = Brain(input_size, hidden_size, output_size)
print(model)