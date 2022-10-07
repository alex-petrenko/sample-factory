from sample_factory.utils.attr_dict import AttrDict


class Configurable:
    def __init__(self, cfg: AttrDict):
        self.cfg: AttrDict = cfg
