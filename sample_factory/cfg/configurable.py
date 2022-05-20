from sample_factory.utils.utils import AttrDict


class Configurable:
    def __init__(self, cfg: AttrDict):
        self.cfg: AttrDict = cfg
