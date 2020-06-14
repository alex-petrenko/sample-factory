ENV_REGISTRY = None


class EnvRegistryEntry:
    def __init__(self, env_name_prefix, make_env_func, add_extra_params_func=None, override_default_params_func=None):
        self.env_name_prefix = env_name_prefix
        self.make_env_func = make_env_func
        self.add_extra_params_func = add_extra_params_func
        self.override_default_params_func = override_default_params_func


def register_env(env_name_prefix, make_env_func, add_extra_params_func=None, override_default_params_func=None):
    """
    For every supported family of environments we require four components:

    :param env_name_prefix: name prefix, e.g. atari_. This allows us to register a single entry per env family
    rather than individual env. Prefix can also, of course, be a full name of the environment.
    :param make_env_func: Factory function that creates an environment instance.
    This function is called like:
    make_my_env(full_env_name, cfg=cfg, env_config=env_config)
    Where full_env_name is a name of the environment to be created, cfg is a namespace with all CLI arguments, and
    env_config is an auxiliary dictionary containing information such as worker index on which the environment lives
    (some envs may require this information)
    :param add_extra_params_func: (optional) function that adds additional parameters to the argument parser.
    This is a very easy way to make your envs configurable through command-line interface.
    :param override_default_params_func: (optional) function that can override the default command line arguments in
    the parser. Every environment demands its own unique set of model architectures and hyperparameters, so this
    mechanism allows us to specify these default parameters once per family of envs to avoid typing them every time we
    want to launch an experiment.

    See the examples for the default envs, it's actually very simple.

    If you want to use a Gym env, just create an empty make_env_func that ignores other parameters and just
    creates a copy of your environment.

    """

    assert callable(make_env_func), 'make_env_func should be callable'
    assert env_name_prefix not in ENV_REGISTRY, f'env name prefix {env_name_prefix} is already in the registry'

    entry = EnvRegistryEntry(env_name_prefix, make_env_func, add_extra_params_func, override_default_params_func)
    ENV_REGISTRY[env_name_prefix] = entry


def ensure_env_registry_initialized():
    global ENV_REGISTRY

    if ENV_REGISTRY is not None:
        return

    ENV_REGISTRY = dict()

    # register default envs
    # gracefully handle import errors so we don't fail when certain environments are not found and we're not using them

    try:
        from envs.doom.doom_utils import make_doom_env
        register_env('doom_', make_doom_env, add_doom_env_args, doom_override_defaults)
    except ImportError:
        pass

    try:
        from envs.atari.atari_utils import make_atari_env
        register_env('atari_', make_atari_env) #TODO
    except ImportError:
        pass

    try:
        from envs.dmlab.dmlab_env import make_dmlab_env
        register_env('dmlab_', make_dmlab_env)
    except ImportError:
        pass

    try:
        from envs.mujoco.mujoco_utils import make_mujoco_env
        register_env('mujoco_', make_mujoco_env)
    except ImportError:
        pass

    try:
        from envs.quadrotors.quad_utils import make_quadrotor_env
        register_env('quadrotor_', make_quadrotor_env)
    except ImportError:
        pass

    try:
        from envs.quadrotors.quad_utils import make_quadrotor_env
        register_env('quadrotor_', make_quadrotor_env)
    except ImportError:
        pass

    try:
        from envs.minigrid.minigrid_utils import make_minigrid_env
        register_env('MiniGrid', make_minigrid_env)
    except ImportError:
        pass
