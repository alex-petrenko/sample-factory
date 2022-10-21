from sample_factory.utils.utils import is_module_available


def envpool_available():
    return is_module_available("envpool")
