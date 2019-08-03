import json

from os.path import join

from utils.utils import log, experiment_dir


# noinspection PyMethodMayBeStatic
class Params:
    def __init__(self, experiment_name, **kwargs):
        self.experiments_root = None

        # internal params, not for CLI
        self._experiment_name = experiment_name

        self._command_line = None
        self._params_serialized = False

    @staticmethod
    def filename_prefix():
        return ''

    def _params_file(self):
        params_filename = self.filename_prefix() + 'params.json'
        return join(self.experiment_dir(), params_filename)

    def experiment_name(self):
        return self._experiment_name

    def experiment_dir(self):
        return experiment_dir(self._experiment_name, self.experiments_root)

    def set_command_line(self, argv):
        self._command_line = ' '.join(argv)

    def ensure_serialized(self):
        if not self._params_serialized:
            self.serialize()
            self._params_serialized = True

    def serialize(self):
        with open(self._params_file(), 'w') as json_file:
            json.dump(self.__dict__, json_file, indent=2)

    def load(self):
        with open(self._params_file()) as json_file:
            json_params = json.load(json_file)
            self.__dict__.update(json_params)
            return self

    @classmethod
    def add_cli_args(cls, parser):
        tmp_params = cls(None)

        def add_arg(argname, type_, default):
            parser.add_argument('--' + argname, type=type_, default=default)

        def str2bool(v):
            return v.lower() == 'true'

        for name, value in tmp_params.__dict__.items():
            # do not add "protected" attributes to CLI
            if name.startswith('_'):
                continue

            if value is None:
                add_arg(name, str, value)
            elif type(value) is bool:
                parser.add_argument('--' + name, type=str2bool, default=str(value))
            elif type(value) in (int, float, str):
                add_arg(name, type(value), value)

    def update(self, args):
        arg_attrs = args.__dict__.keys()

        for name, value in self.__dict__.items():
            if name in arg_attrs:
                new_value = args.__dict__[name]
                if value != new_value:
                    log.info('Replacing default value for %s (%r) with new value: %r', name, value, new_value)
                    setattr(self, name, new_value)
