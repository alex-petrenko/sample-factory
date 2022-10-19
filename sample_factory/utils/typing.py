from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from gym import spaces

from sample_factory.utils.attr_dict import AttrDict

Config = Union[argparse.Namespace, AttrDict]

StatusCode = int

PolicyID = int
Device = str

# these can be fake wrapper classes if we're in serial mode, so using Any
MpQueue = Any
MpLock = Any

Env = Any
ObsSpace = Union[spaces.Space, spaces.Dict]
ActionSpace = spaces.Space

CreateEnvFunc = Callable[[str, Optional[Config], Optional[AttrDict], Optional[str]], Env]

# there currenly isn't a single class all distributions derive from, but we gotta use something for the type hint
ActionDistribution = Any

InitModelData = Tuple[PolicyID, Dict, torch.device, int]
