from typing import Tuple

import torch

from models.base import UncertaintyModel
from .base import UncertaintyMixin
import helpers.functional as F


class MonteCarloMixin(UncertaintyMixin):
    def __init__(self, **kwargs):
        assert isinstance(self, UncertaintyModel)
        self.num_samples = kwargs.get('num_samples', 10)

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        F.enable_dropout(self)
        predictions = []
        for i in range(self.num_samples):
            predictions.append(self.forward(*args, *kwargs))
        # TODO Monte Carlo estimation
