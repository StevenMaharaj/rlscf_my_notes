from typing import Generic, Callable
from abc import ABC
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
# import graphviz
import numpy as np
from pprint import pprint
from typing import (Callable, Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, TypeVar, Set)

from distribution import (Categorical, Distribution, FiniteDistribution,
                          SampledDistribution)

S = TypeVar('S')
X = TypeVar('X')


S = TypeVar('S')
X = TypeVar('X')


class State(ABC, Generic[S]):
    state: S

    # def on_non_terminal(self,f: Callable[[NonTerminal[S]], X],default: X) -> X:
    #     if isinstance(self, NonTerminal):
    #         return f(self)
    #     else:
    #         return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S


class MarkovProcess(ABC, Generic[S]):
    '''A Markov process with states of type S.
    '''
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        '''Given a state of the process, returns a distribution of
        the next states.  Returning None means we are in a terminal state.
        '''

    def simulate(
        self,
        start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[State[S]]:
        '''Run a simulation trace of this Markov process, generating the
        states visited during the trace.
        This yields the start state first, then continues yielding
        subsequent states forever or until we hit a terminal state.
        '''

        state: State[S] = start_state_distribution.sample()
        yield state

        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state

    def traces(
            self,
            start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[Iterable[State[S]]]:
        '''Yield simulation traces (the output of `simulate'), sampling a
        start state from the given distribution each time.
        '''
        while True:
            yield self.simulate(start_state_distribution)


Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]
