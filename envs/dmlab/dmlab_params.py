def dmlab_override_default_params_and_args(params, args):
    params.obs_subtract_mean = 128.0
    params.obs_scale = 128.0

    params.conv_filters = [
        [3, 32, 8, 4],
        [32, 64, 4, 2],
        [64, 128, 3, 2],
    ]

    params.hidden_size = 512

    # TODO: put into params. This is a part of the algorithm after all, not the part of the environment
    if 'render_action_repeat' in args:
        if args.render_action_repeat is None:
            args.render_action_repeat = 4
