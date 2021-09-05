from typing import Sequence, TypeVar
from typing import Set, FiniteDistribution, Categorical, Mapping

from rl.markov_process import MarkovProcess, NonTerminal, Terminal, State

S = TypeVar("S")

Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]


class FiniteMarkovProcess(MarkovProcess[S]):
    non_terminal_state: Sequence[NonTerminal[S]]
    transition_map: Transition[S]


def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
    non_terminals: Set[S] = set(transition_map.keys())
    self.transition_map = {
        NonTerminal(s): Categorical(
            {(NonTerminal(s1) if s1 in non_terminals else Terminal(s1)): p
             for s1, p in v.table().items()}
        ) for s, v in transition_map.items()
    }
    self.non_terminal_states = list(self.transition_map.keys())


def __repr__(self) -> str:
    display = ""
    for s, d in self.transition_map.items():
        display += f"From State {s.state}:\n"
        for s1, p in d:
            opt = "Terminal " if isinstance(s1, Terminal) else ""
            display += f" To {opt}State {s1.state} with Probability {p:.3f}\n"
    return display

    def transition(self, state: NonTerminal[S])\
            -> FiniteDistribution[State[S]]:
        return self.transition_map[state]
