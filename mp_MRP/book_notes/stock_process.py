import numpy as np

from dataclasses import  dataclass

@dataclass
class Process1:
    @dataclass
    class State:
        price: int
    
    level_param: int
    alpha1: float = 0.25

    def up_prob(self,state: State) -> float:
        return 1. / (1 + np.exp(-self.alpha1*(self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state),1)[0]
        return Process1.State(price = state.price + up_move*2 - 1)

def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)