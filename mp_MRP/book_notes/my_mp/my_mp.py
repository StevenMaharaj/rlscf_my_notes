
from __future__ import annotations
from typing import (Callable, Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, TypeVar, Set)

from dataclasses import dataclass
from abc import ABC, abstractmethod

from rl.distribution import Distribution


S = TypeVar('S')
X = TypeVar('X')


class State(ABC, Generic[S]):
    state: S

    def on_non_terminal(
        self,
        f: Callable[[NonTerminal[S]], X],
        default: X
    ) -> X:
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S


class MarkovProcess(ABC,Generic[S]):

    @abstractmethod
    def transition(self,state: NonTerminal[S]) -> Distribution[State[S]]:
        pass

    def simulate(self,start_state_distribution: Distribution[NonTerminal[S]]) -> Iterable[State[S]]:

        # starting state
        state: State[S] = start_state_distribution.sample()
        yield State

        # yeild values until you get to a Terminal State
        while isinstance(state,NonTerminal):
            state = self.transition(state).sample()
            yield state
