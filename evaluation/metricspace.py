from enum import Enum
from evaluation.metrics import boundary_f_measure, jaccard_index, panoptic_quality

class MetricName(Enum):
    PQ = "pq"  # Panoptic Quality
    BF1M = "bf1m" # Boundary F1 Score
    MIOU = "miou" # Mean IoU


_METRIC_FACTORY = {
    MetricName.PQ: panoptic_quality,
    MetricName.BF1M: boundary_f_measure,
    MetricName.MIOU: jaccard_index
}
if __name__ == "__main__":
    pass
# print(_METRIC_FACTORY["0"])