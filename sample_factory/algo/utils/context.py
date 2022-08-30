from typing import Callable, Dict

from sample_factory.model.model_factory import ModelFactory


class SampleFactoryContext:
    def __init__(self):
        self.env_registry = dict()
        self.model_factory = ModelFactory()


GLOBAL_CONTEXT = None


def sf_global_context() -> SampleFactoryContext:
    global GLOBAL_CONTEXT
    if GLOBAL_CONTEXT is None:
        GLOBAL_CONTEXT = SampleFactoryContext()
    return GLOBAL_CONTEXT


def set_global_context(ctx: SampleFactoryContext):
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = ctx


def global_env_registry() -> Dict[str, Callable]:
    """
    :return: global env registry
    :rtype: EnvRegistry
    """
    return sf_global_context().env_registry


def global_model_factory() -> ModelFactory:
    """
    :return: global model factory
    :rtype: ModelFactory
    """
    return sf_global_context().model_factory
