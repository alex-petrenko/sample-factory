import os
from os.path import join

import numpy as np

from collections import OrderedDict

from sample_factory.utils.utils import ensure_dir_exists


class ParamGenerator:
    def __init__(self):
        pass

    def generate_params(self, randomize=True):
        """Supposed to be a generator (so should yield dicts of parameters)."""
        pass


class ParamList(ParamGenerator):
    """The most simple kind of generator, represents just the list of parameter combinations."""

    def __init__(self, combinations):
        super(ParamList, self).__init__()
        self.combinations = combinations

    def generate_params(self, randomize=True):
        if randomize:
            combinations = np.random.permutation(self.combinations)
        else:
            combinations = self.combinations

        for combination in combinations:
            yield combination


class ParamGrid(ParamGenerator):
    """Parameter generator for grid search."""

    def __init__(self, grid_tuples):
        """Uses OrderedDict, so must be initialized with the list of tuples if you want to preserve order."""
        super(ParamGrid, self).__init__()
        self.grid = OrderedDict(grid_tuples)

    def _generate_combinations(self, param_idx, params):
        """Recursively generate all parameter combinations in a grid."""

        if param_idx == len(self.grid) - 1:
            # last parameter, just return list of values for this parameter
            return [[value] for value in self.grid[params[param_idx]]]
        else:
            subcombinations = self._generate_combinations(param_idx + 1, params)  # returns list of param combinations
            result = []

            # iterate over all values of current parameter
            for value in self.grid[params[param_idx]]:
                for subcombination in subcombinations:
                    result.append([value] + subcombination)

            return result

    def generate_params(self, randomize=True):
        if len(self.grid) == 0:
            return dict()

        # start with 0th value for every parameter
        total_num_combinations = np.prod([len(p_values) for p_values in self.grid.values()])

        param_names = tuple(self.grid.keys())
        all_combinations = self._generate_combinations(0, param_names)

        assert len(all_combinations) == total_num_combinations

        if randomize:
            all_combinations = np.random.permutation(all_combinations)

        for combination in all_combinations:
            combination_dict = {param_name: combination[i] for (i, param_name) in enumerate(param_names)}
            yield combination_dict


class Experiment:
    def __init__(self, name, cmd, param_generator, env_vars=None):
        """
        :param cmd: base command to append the parameters to
        :param param_generator: iterable of parameter dicts
        """
        self.base_name = name
        self.cmd = cmd
        self.params = list(param_generator)
        self.env_vars = env_vars

    def generate_experiments(self):
        """Yields tuples of (cmd, experiment_name)"""
        num_experiments = 1 if len(self.params) == 0 else len(self.params)

        for experiment_idx in range(num_experiments):
            cmd_tokens = [self.cmd]
            experiment_name_tokens = [self.base_name]

            # abbreviations for parameter names that we've used
            param_abbrs = []

            if len(self.params) > 0:
                params = self.params[experiment_idx]
                for param, value in params.items():
                    param_str = f'--{param} {value}'
                    cmd_tokens.append(param_str)

                    abbr = None
                    for l in range(len(param)):
                        abbr = param[:l+3]
                        if abbr not in param_abbrs:
                            break

                    param_abbrs.append(abbr)
                    experiment_name_token = f'{abbr}_{value}'
                    experiment_name_tokens.append(experiment_name_token)

            experiment_name = f'{experiment_idx:02d}_' + '_'.join(experiment_name_tokens)

            cmd_tokens.append(f'--experiment {experiment_name}')
            param_str = ' '.join(cmd_tokens)

            yield param_str, experiment_name


class RunDescription:
    def __init__(self, run_name, experiments, train_dir=None):
        if train_dir is None:
            train_dir = ensure_dir_exists(join(os.getcwd(), 'train_dir'))

        self.train_dir = train_dir
        self.run_name = run_name
        self.experiments = experiments
        self.experiment_suffix = ''

    def generate_experiments(self):
        """Yields tuples (final cmd for experiment, experiment_name, root_dir)."""
        for experiment in self.experiments:
            root_dir = join(self.run_name, f'{experiment.base_name}_{self.experiment_suffix}')

            experiment_cmds = experiment.generate_experiments()
            for experiment_cmd, experiment_name in experiment_cmds:
                experiment_cmd += f' --train_dir={self.train_dir} --experiments_root={root_dir}'
                yield experiment_cmd, experiment_name, root_dir, experiment.env_vars
