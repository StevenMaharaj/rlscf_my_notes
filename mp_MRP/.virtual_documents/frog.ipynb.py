import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from scipy.sparse.linalg import spsolve

from numpy.linalg import solve, norm

from numpy.random import rand
from time import sleep
import matplotlib.pyplot as plt
plt.style.use('ggplot')


n = 10
Transitions_matrix = lil_matrix((n,n+1))

for row in range(n):
    for col in range(n+1):
        if row<col:
            prob = 1/(n - row)
            Transitions_matrix[row,col] = prob

# Transitions_matrix[n,n] = 1


# Transitions_matrix.toarray()


def simulate_game(Transitions_matrix):
    n = Transitions_matrix.shape[1]
    terminal_state = Transitions_matrix.shape[0]
    s = 0
    t = 0
    all_states = np.arange(n)
    while s get_ipython().getoutput("= terminal_state:")
        s = np.random.choice(all_states,p=Transitions_matrix[s].toarray()[0])
        t += 1
    return t


simulate_game(Transitions_matrix)


n_simulations = 10000
J = []
for _ in range(n_simulations):
    J.append(simulate_game(Transitions_matrix))


np.array(J).mean()


11/6



