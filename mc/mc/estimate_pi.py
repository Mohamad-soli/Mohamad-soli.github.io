import numpy as np
import matplotlib.pyplot as plt

# throwing the darts
no_of_points = 1_000
x = np.random.uniform(-1, 1, no_of_points)
y = np.random.uniform(-1, 1, no_of_points)
# counting the number of darts that land inside the circle
inside = x**2 + y**2 <= 1
# estimating pi
pi = 4 * inside.sum() / no_of_points
print(pi)
# Visualizing the results
fig, ax = plt.subplots()
ax.set_aspect('equal')
# matplotlib add patch is a very handy tool for drawing shapes inside a plot
ax.add_patch(plt.Circle((0, 0), 1, color='r', fill=False))
ax.add_patch(plt.Rectangle((-1, -1), 2, 2, color='b', fill=False))

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off') 
plt.savefig('blog_posts/mc/square_circle.png')
# different colors for the points inside and outside of the circle
ax.scatter(x[inside], y[inside], color='r', s=1)
ax.scatter(x[~inside], y[~inside], color='b', s=1)
plt.savefig('blog_posts/mc/darts_flying.png')
plt.show()