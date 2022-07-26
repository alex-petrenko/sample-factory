import hashlib

from sample_factory.utils.utils import is_module_available


def dmlab_available():
    return is_module_available("deepmind_lab")


def string_to_hash_bucket(s, vocabulary_size):
    return (int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (vocabulary_size - 1)) + 1
