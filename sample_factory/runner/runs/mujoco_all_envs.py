from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['mujoco_ant', 'mujoco_halfcheetah', 'mujoco_hopper', 'mujoco_humanoid', 'mujoco_doublependulum', 'mujoco_pendulum', 'mujoco_reacher', 'mujoco_swimmer', 'mujoco_walker']),
])

_experiments = [
    Experiment(
        'mujoco_all_envs',
        'python -m sample_factory_examples.mujoco_examples.train_mujoco --algo=APPO',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('mujoco_all_envs', experiments=_experiments)
