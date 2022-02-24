from .qrlepso_base import QrlepsoBaseSwarm
import numpy as np


class I8Rlepso(QrlepsoBaseSwarm):
    optimizer_name = 'I8-RLEPSO'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'I8-RLEPSO'
        self.quantization = 8

    def data_quantization(self, data, data_range):
        quantization_range = 2 ** (self.quantization - 1)
        data = np.round(data / data_range * quantization_range) / quantization_range * data_range
        return data
