from typing import Callable, Dict


class SampleFactoryContext:
    def __init__(self):
        self.env_registry = dict()
        self.encoder_registry = dict()


GLOBAL_CONTEXT = None


def global_env_registry() -> Dict[str, Callable]:
    """
    :return: global env registry
    :rtype: EnvRegistry
    """
    return sf_global_context().env_registry


def global_encoder_registry():
    return sf_global_context().encoder_registry


def sf_global_context() -> SampleFactoryContext:
    global GLOBAL_CONTEXT
    if GLOBAL_CONTEXT is None:
        GLOBAL_CONTEXT = SampleFactoryContext()
    return GLOBAL_CONTEXT


def set_global_context(ctx: SampleFactoryContext):
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = ctx
