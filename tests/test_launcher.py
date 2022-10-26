import logging
import shutil
from os.path import join, split

import numpy as np

from sample_factory.launcher.run import launcher_argparser
from sample_factory.launcher.run_description import Experiment, ParamGrid, ParamList, RunDescription
from sample_factory.launcher.run_processes import run
from sample_factory.utils.utils import ensure_dir_exists, project_tmp_dir


class TestParams:
    def test_param_list(self):
        params = [
            {"p1": 1, "p2": "a"},
            {"p2": "b", "p4": "test"},
        ]
        param_list = ParamList(params)
        param_combinations = list(param_list.generate_params(randomize=False))

        for i, combination in enumerate(params):
            assert combination == param_combinations[i]

    def test_param_grid(self):
        grid = ParamGrid(
            [
                ("p1", [0, 1]),
                ("p2", ["a", "b", "c"]),
                ("p3", [None, {}]),
            ]
        )
        param_combinations = grid.generate_params(randomize=True)
        for p in param_combinations:
            for key in ("p1", "p2", "p3"):
                assert key in p

        param_combinations = list(grid.generate_params(randomize=False))
        assert param_combinations[0] == {"p1": 0, "p2": "a", "p3": None}
        assert param_combinations[1] == {"p1": 0, "p2": "a", "p3": {}}
        assert param_combinations[-2] == {"p1": 1, "p2": "c", "p3": None}
        assert param_combinations[-1] == {"p1": 1, "p2": "c", "p3": {}}


class TestLauncher:
    def test_experiment(self):
        params = ParamGrid([("p1", [3.14, 2.71]), ("p2", ["a", "b", "c"])])
        cmd = "python super_rl.py"
        ex = Experiment("test", cmd, params.generate_params(randomize=False))
        cmds = ex.generate_experiments("train_dir", customize_experiment_name=True, param_prefix="--")
        for index, value in enumerate(cmds):
            command, name = value
            assert command.startswith(cmd)
            assert name.startswith(f"0{index}_test")

    def test_descr(self):
        params = ParamGrid([("p1", [3.14, 2.71]), ("p2", ["a", "b", "c"])])
        experiments = [
            Experiment("test1", "python super_rl1.py", params.generate_params(randomize=False)),
            Experiment("test2", "python super_rl2.py", params.generate_params(randomize=False)),
        ]
        rd = RunDescription("test_run", experiments)
        cmds = rd.generate_experiments("train_dir")
        for command, name, root_dir, env_vars in cmds:
            exp_name = split(root_dir)[-1]
            assert "--experiment" in command
            assert exp_name in name
            assert root_dir.startswith("test_run")

    def test_simple_cmd(self):
        logging.disable(logging.INFO)

        echo_params = ParamGrid(
            [
                ("p1", [3.14, 2.71]),
                ("p2", ["a", "b", "c"]),
                ("p3", list(np.arange(3))),
            ]
        )
        experiments = [
            Experiment("test_echo1", "echo", echo_params.generate_params(randomize=True)),
            Experiment("test_echo2", "echo", echo_params.generate_params(randomize=False)),
        ]
        train_dir = ensure_dir_exists(join(project_tmp_dir(), "tests"))
        root_dir_name = "__test_run__"
        rd = RunDescription(root_dir_name, experiments)

        args = launcher_argparser([]).parse_args([])
        args.max_parallel = 8
        args.pause_between = 0
        args.train_dir = train_dir

        run(rd, args)

        rd2 = RunDescription(
            root_dir_name,
            experiments,
            experiment_arg_name="--experiment_tst",
            experiment_dir_arg_name="--dir",
        )
        run(rd2, args)

        logging.disable(logging.NOTSET)

        shutil.rmtree(join(train_dir, root_dir_name))
