from __future__ import annotations

import argparse
from typing import Any, Callable, Optional, Union

from sample_factory.utils.utils import AttrDict

Config = Union[argparse.Namespace, AttrDict]

StatusCode = int

PolicyID = int
Device = str

# these can be fake wrapper classes if we're in serial mode, so using Any
MpQueue = Any
MpLock = Any

# maybe replace with a proper type hint
Env = Any

CreateEnvFunc = Callable[[str, Optional[Config], Optional[AttrDict]], Env]
