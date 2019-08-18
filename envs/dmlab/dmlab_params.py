def dmlab_override_defaults(env, parser):
    """Currently use same parameters as Doom."""
    from envs.doom.doom_params import doom_override_defaults
    doom_override_defaults(env, parser)
