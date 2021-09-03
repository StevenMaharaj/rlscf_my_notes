from __future__ import annotations

from rl.distribution import Categorical
from rl.gen_utils.common_funcs import get_unit_sigmoid_func

from typing import (Callable, Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, TypeVar, Set)

from dataclasses import dataclass
from abc import ABC, abstractmethod

from my_mp import *

@dataclass
class StateMP3:
    num_up_moves: int
    num_down_moves: int

@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):
    alpha3: float  = 1.0
# if total = 0 set it to 0.5
    def up_prob(self, state: StateMP3) -> float:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3) (
            state.num_down_moves / total
        ) if total else 0.5
