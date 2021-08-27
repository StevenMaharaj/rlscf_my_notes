import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from scipy.sparse.linalg import spsolve

from numpy.linalg import solve, norm

from numpy.random import rand
from time import sleep
import matplotlib.pyplot as plt
plt.style.use('ggplot')


snakes_ladders_map = [(2,23),(8,34),(20,77),(32,68),(41,79),(74,88),(85,95),(82,100),
                     (29,9),(38,15),(47,5),(53,33),(86,54),(97,25),(92,70)] 
snakes_ladders_map = np.array(snakes_ladders_map) - 1
snakes_ladders_map_arg = snakes_ladders_map[:,0]
# snakes_ladders_map


n = 100
Transitions_matrix = lil_matrix((n,n))
for row in range(n):
    for col in range(1,7):
        if row+col > 99:
            break
        else:
            Transitions_matrix[row,row+col] = 1/6
for el in snakes_ladders_map:
    for i in range(1,7):
        row,col = el[0],el[1]
        if row - i < 0:
            break
        else:
            Transitions_matrix[row - i,row] = 0.0
            Transitions_matrix[row - i,col] = 1/6
for row in range(94,99):
    self_prob = (6 - (99 - row))/6
    Transitions_matrix[row,row] = self_prob
Transitions_matrix[99,99] = 1


Transitions_matrix[99].toarray()


def simulate_game():
    n = 100
    terminal_state = 99
    s = 0
    t = 0
    all_states = np.arange(n)
    while s get_ipython().getoutput("= terminal_state:")
        s = np.random.choice(all_states,p=Transitions_matrix[s].toarray()[0])
        t += 1
    return t
        
        
    


n_simulations = 10000
T = []
for _ in range(n_simulations):
    T.append(simulate_game())



plt.hist(T,bins= 100)
plt.show()


np.array(T).mean()



Transitions_matrix[0,:].toarray()[0]


all_states



