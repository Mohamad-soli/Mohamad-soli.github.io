import torch
import torch.nn as nn

class Rosenbrock(nn.Module):
    def __init__(self, a, b):
        super(Rosenbrock, self).__init__()
        # Initializing the Rosenbrock function
        self.a = a
        self.b = b
        # Optimization parameters are randomly initialized and
        # defined to be a nn.Parameter object.
    
    def forward(self, x, y):
        # Here is the function that is being optimized
        return (x - self.a) ** 2 + self.b * (y - x ** 2) ** 2
