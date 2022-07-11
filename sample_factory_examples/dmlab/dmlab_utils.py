import hashlib

# custom experiments can define functions to this list to do something extra with the raw episode summaries
# coming from the environments
EXTRA_EPISODIC_STATS_PROCESSING = []

# custom experiments or environments can append functions to this list to postprocess some summaries, or aggregate
# summaries, or do whatever else the user wants
EXTRA_PER_POLICY_SUMMARIES = []


def string_to_hash_bucket(s, vocabulary_size):
    return (int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (vocabulary_size - 1)) + 1
