import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

global iter
iter = 0

class Rosenbrock(nn.Module):
    def __init__(self, a, b):
        super(Rosenbrock, self).__init__()
        # Initializing the Rosenbrock function
        self.a = a
        self.b = b
        # Optimization parameters are randomly initialized and
        # defined to be a nn.Parameter object.
        self.x = torch.nn.Parameter(torch.Tensor([-1.0]))
        self.y = torch.nn.Parameter(torch.Tensor([2.0]))
    
    def forward(self,):
        # Here is the function that is being optimized
        return (self.x - self.a) ** 2 + self.b * (self.y - self.x ** 2) ** 2

    def optimizer_setup(self, model):
        # Initializing the optimizer
        self.optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter = 5000)
        self.optimizer.step(self.closure)
        # The final optimized parameters are returned
        print(f"The Optimized Value for X: {self.x.item()} and Y: {self.y.item()}")

    def closure(self, ):
        self.optimizer.zero_grad()
        # we set the loss to be the difference from being Zero
        loss = self.forward()
        loss.backward()
        print(f"Loss: {loss.item()}  X={self.x.item()} Y={self.y.item()}")
        self.plot()
        return loss

    def plot(self,):
        global iter
        plt.figure()
        x = np.linspace(-2, 2, 1000)
        y = np.linspace(-1, 3, 1000)
        X, Y = np.meshgrid(x, y)
        Z = (X - self.a) ** 2 + self.b * (Y - X ** 2) ** 2
        plt.contour(X, Y, Z, levels = [0, 1, 10, 100, 400], cmap='gray')
        plt.scatter(self.x.item(), self.y.item())
        plt.savefig(f"blog_posts/anim/lr001/rosenbrock_plot_lr=001_{iter}.png")
        iter += 1
        plt.close('all')

if __name__ == "__main__":
    # Initializing the Rosenbrock function
    func1 = Rosenbrock(1, 100)
    func1.optimizer_setup(func1)
