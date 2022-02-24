from .qrlepso_base import QrlepsoBaseSwarm
import numpy as np

class F16Rlepso(QrlepsoBaseSwarm):
    optimizer_name = 'F16-RLEPSO'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'F16-RLEPSO'

    def data_quantization(self, data, data_range):
        return np.float16(data)
