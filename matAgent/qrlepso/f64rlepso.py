from .qrlepso_base import QrlepsoBaseSwarm


class F64Rlepso(QrlepsoBaseSwarm):

    optimizer_name = 'F64-RLEPSO'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'F64-RLEPSO'

    def data_quantization(self, data, data_range):
        return data
