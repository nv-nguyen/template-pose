from easydict import EasyDict as edict
import json
import os


class Config(object):
    """
    config: json ---> edict
    """

    def __init__(self, config_file=None):
        assert os.path.exists(config_file), 'config file is not existed.'
        self.config_file = config_file
        self.load()

    def load(self):
        with open(self.config_file, "r") as fd:
            self.config_json = json.load(fd)
            self.config = edict(self.config_json)

    def get_config(self):
        return self.config
