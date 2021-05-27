import hashlib


def string_to_hash_bucket(s, vocabulary_size):
    return (int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % (vocabulary_size - 1)) + 1
