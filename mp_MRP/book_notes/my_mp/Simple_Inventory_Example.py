from rl.distribution import Categorical, FiniteDistribution
from scipy.stats import poisson
from rl.markov_process import MarkovProcess,FiniteMarkovProcess

from typing import Mapping, Dict
from typing import Sequence


from dataclasses import dataclass









@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int 


    def inventory_position(self) -> int:
        return self.on_hand + self.on_order

class SimpleInventoryMPFinite(FiniteMarkovProcess[InventoryState]):

    def __init__(self,capacity:int,poisson_lambda: float):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.poisson_distr = poisson(poisson_lambda)
        # super() acts like self but represents the base class
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Mapping[InventoryState,FiniteDistribution[InventoryState]]:
        d: dict[InventoryState,Categorical[InventoryState]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha,beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                state_probs_map: Mapping[InventoryState,float] = {
                    InventoryState(ip - 1,beta1):
                    (self.poisson_distr.pmf(i) if i < ip else 
                    1- self.poisson_distr.cdf(ip - 1))
                    for i in range(ip + 1)
                }
                d[InventoryState(alpha,beta)] = Categorical(state_probs_map)
        return d

user_capacity = 2
user_poisson_lambda = 1.0

si_mp = SimpleInventoryMPFinite(capacity=user_capacity,
poisson_lambda=user_poisson_lambda)

print(si_mp)