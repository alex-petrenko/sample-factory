"""
Utilities for sharing model parameters between components.
"""
import sys
from typing import Optional

import torch
from torch import Tensor

from sample_factory.algo.utils.multiprocessing_utils import get_lock, get_mp_ctx
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log


class ParameterServer:
    def __init__(self, policy_id, policy_versions: Tensor, serial_mode: bool):
        self.policy_id = policy_id
        self.actor_critic = None
        self.policy_versions = policy_versions
        self.device: Optional[torch.device] = None

        mp_ctx = get_mp_ctx(serial_mode)
        self._policy_lock = get_lock(serial_mode, mp_ctx)

    @property
    def policy_lock(self):
        return self._policy_lock

    def init(self, actor_critic, policy_version, device: torch.device):
        self.actor_critic = actor_critic
        self.policy_versions[self.policy_id] = policy_version
        self.device = device
        log.debug("Initialized policy %d weights for model version %d", self.policy_id, policy_version)

    def update_weights(self, policy_version):
        """
        In async algorithms policy_versions tensor is in shared memory.
        Therefore clients can just look at the location in shared memory once in a while to see if the
        weights are updated.
        """
        self.policy_versions[self.policy_id] = policy_version


class ParameterClient:
    def __init__(self, param_server: ParameterServer, cfg, env_info, timing: Timing):
        self.server = param_server
        self.policy_id = param_server.policy_id
        self.policy_versions = param_server.policy_versions

        self.cfg = cfg
        self.env_info = env_info

        self.latest_policy_version = -1

        self._actor_critic = None
        self._policy_lock = param_server.policy_lock

        self.timing = timing

    @property
    def actor_critic(self):
        return self._actor_critic

    @property
    def policy_version(self):
        return self.latest_policy_version

    def _get_server_policy_version(self):
        return self.policy_versions[self.policy_id].item()

    def on_weights_initialized(self, state_dict, device: torch.device, policy_version: int) -> None:
        self.latest_policy_version = policy_version

    def ensure_weights_updated(self):
        raise NotImplementedError()

    def cleanup(self):
        pass


class ParameterClientSerial(ParameterClient):
    def on_weights_initialized(self, state_dict, device: torch.device, policy_version: int) -> None:
        """
        Literally just save the reference to actor critic since we're in the same process.
        Model should be fully initialized at this point.
        """
        super().on_weights_initialized(state_dict, device, policy_version)
        self._actor_critic = self.server.actor_critic

    def ensure_weights_updated(self):
        """In serial case we don't need to do anything."""
        self.latest_policy_version = self._get_server_policy_version()


class ParameterClientAsync(ParameterClient):
    def __init__(self, param_server: ParameterServer, cfg, env_info, timing: Timing):
        super().__init__(param_server, cfg, env_info, timing)
        self._shared_model_weights = None
        self.num_policy_updates = 0

    @property
    def actor_critic(self):
        assert self.latest_policy_version >= 0, "Trying to access actor critic before it is initialized"
        return self._actor_critic

    def _init_local_copy(self, device, cfg, obs_space, action_space):
        self._actor_critic = create_actor_critic(cfg, obs_space, action_space)
        self._actor_critic.model_to_device(device)

        for p in self._actor_critic.parameters():
            p.requires_grad = False  # we don't train anything here
        self._actor_critic.eval()

    def on_weights_initialized(self, state_dict, device: torch.device, policy_version: int) -> None:
        super().on_weights_initialized(state_dict, device, policy_version)

        self._init_local_copy(device, self.cfg, self.env_info.obs_space, self.env_info.action_space)

        with self._policy_lock:
            if state_dict is None:
                log.warning(f"Parameter client {self.policy_id} received empty state dict, using random weights...")
            else:
                self._actor_critic.load_state_dict(state_dict)
                self._shared_model_weights = state_dict

    def ensure_weights_updated(self):
        server_policy_version = self._get_server_policy_version()
        if self.latest_policy_version < server_policy_version and self._shared_model_weights is not None:
            with self.timing.time_avg("weight_update"), self._policy_lock:
                self._actor_critic.load_state_dict(self._shared_model_weights)

            self.latest_policy_version = server_policy_version

            self.num_policy_updates += 1
            if self.num_policy_updates % 10 == 0:
                log.info(
                    "Updated weights for policy %d, policy_version %d (%s)",
                    self.policy_id,
                    self.latest_policy_version,
                    str(self.timing.weight_update),
                )

    def cleanup(self):
        # TODO: fix termination problems related to shared CUDA tensors (they are harmless but annoying)
        weights = self._shared_model_weights
        del self._actor_critic
        del self._shared_model_weights
        del self.policy_versions

        if weights is not None:
            import gc

            weights_referrers = gc.get_referrers(weights)
            log.debug(f"Weights refcount: {sys.getrefcount(weights)} {len(weights_referrers)}")


def make_parameter_client(is_serial_mode, parameter_server, cfg, env_info, timing: Timing) -> ParameterClient:
    """Parameter client factory."""
    cls = ParameterClientSerial if is_serial_mode else ParameterClientAsync
    return cls(parameter_server, cfg, env_info, timing)
