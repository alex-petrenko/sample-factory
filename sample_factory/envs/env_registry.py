from sample_factory.utils.utils import log


class EnvRegistryEntry:
    def __init__(self, env_name_prefix, make_env_func, add_extra_params_func=None, override_default_params_func=None):
        self.env_name_prefix = env_name_prefix
        self.make_env_func = make_env_func
        self.add_extra_params_func = add_extra_params_func
        self.override_default_params_func = override_default_params_func


class EnvRegistry:
    def __init__(self):
        self.registry = dict()

    def register_env(
        self,
        env_name_prefix,
        make_env_func,
        add_extra_params_func=None,
        override_default_params_func=None,
    ):
        """
        A standard thing to do in RL frameworks is to just rely on unique environment names registered in Gym.
        SampleFactory supports a mechanism on top of that, we define "environment families", e.g. "atari", or "doom",
        and certain things can be defined per env family rather than for specific environment or experiment (such as
        default hyperparameters and env command line arguments).

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

        See the sample_factory_examples for the default envs, it's actually very simple.

        If you want to use a Gym env, just create an empty make_env_func that ignores other parameters and
        instantiates a copy of your Gym environment.

        """

        assert callable(make_env_func), "make_env_func should be callable"

        entry = EnvRegistryEntry(env_name_prefix, make_env_func, add_extra_params_func, override_default_params_func)
        self.registry[env_name_prefix] = entry

        log.debug("Env registry entry created: %s", env_name_prefix)

    def register_env_deferred(self, env_name_prefix, register_env_family_func):
        """Same as register_env but we defer the creation of the registry entry until we actually need it."""
        assert callable(register_env_family_func)

        self.registry[env_name_prefix] = register_env_family_func

    def resolve_env_name(self, full_env_name):
        """
        :param full_env_name: complete name of the environment, to be passed to the make_env_func, e.g. atari_breakout
        :return: env registry entry
        :rtype: EnvRegistryEntry
        """
        # we find a match with a reqgistered env family prefix
        for env_prefix, registry_entry in self.registry.items():
            if not full_env_name.startswith(env_prefix):
                continue

            # We found a match. If it's a callable, we should first handle a deferred registry entry
            if callable(registry_entry):
                make_env_func, add_extra_params_func, override_default_params_func = registry_entry()
                self.register_env(env_prefix, make_env_func, add_extra_params_func, override_default_params_func)

            return self.registry[env_prefix]

        msg = (
            f"Could not resolve {full_env_name}."
            f"Did you register the family of environments in the registry? See sample_factory_examples for details."
        )
        log.warning(msg)
        raise RuntimeError(msg)
