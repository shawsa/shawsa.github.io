import numpy as np
from itertools import product

x_values = [1, 2, 3]
y_values = [4, 5, 6]
#  loop over collections simultaneously
for x, y in zip(x_values, y_values):
    pass


alphas = [0, 0.1, 0.2]
betas = [0, 3, 6]
# Loop over every pair in the Cartesian Product
for alpha, beta in product(alphas, betas):
    pass

# do something with the solutions of a PDE 
for time, u, v, w in solver.solution_generator():
    pass


# loop over states of a Markov chain
for index, state in enumerate(markov_chain):
    pass
