from dataclasses import  dataclass
from abc import ABC,abstractmethod
from typing import Generic, TypeVar,Iterable
from dist import Dist

A = TypeVar('A')
S = TypeVar('S')

class State:
    pass

class NTState:
    pass

class TState:
    pass




class Policy(ABC,Generic[S,A]):
    @abstractmethod
    def act(self, state: S):
        pass