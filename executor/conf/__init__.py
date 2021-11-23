# -*- coding: utf-8 -*-n
import os

from executor.conf.env import EnvWrapper

ENV = EnvWrapper(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), True, 'TrainingExecutor')
