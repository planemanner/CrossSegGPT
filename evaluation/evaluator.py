"""
Wrapping Instance 구현
"""
from metricspace import _METRIC_FACTORY, MetricName
from typing import Iterable, Dict
from utils import AverageMeter

class Evaluator:
    def __init__(self):
        self._meters = {}

    def register_metric(self, *names: Iterable[str]):
        for name in names:
            if isinstance(name, str):
                try:
                    name = MetricName(name.lower())
                except ValueError:
                    raise ValueError(f"Unsupported metric: {name}")
            if name not in _METRIC_FACTORY:
                raise ValueError(f"Metric {name.value} not implemented.")
            
            self._meters[name] = AverageMeter(_METRIC_FACTORY[name])
        return self
    
    def update(self, pred, gt, **kwargs):
        for meter in self._meters.values():
            meter.update(pred, gt, **kwargs)

    def compute(self) -> Dict[str, float]:
        return {name.value: meter.compute() for name, meter in self._meters.items()}

    def reset(self):
        for meter in self._meters.values():
            meter.reset()    

if __name__ == "__main__":
    evaler = Evaluator()
    evaler.register_metric('pq')
