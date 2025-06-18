from typing import Callable
from abc import ABC, abstractmethod

class Meter(ABC):
    @abstractmethod
    def update(self, pred, gt, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

class AverageMeter(Meter):
    def __init__(self, metric_fn: Callable):
        self.metric_fn = metric_fn
        self.values = []
    
    def update(self, pred, gt, **kwargs):
        value = self.metric_fn(pred, gt, **kwargs)
        self.values.append(value)

    def compute(self):
        return sum(self.values) / len(self.values) if self.values else 0.0

    def reset(self):
        self.values = []