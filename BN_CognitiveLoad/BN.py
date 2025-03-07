import sys
import pandas as pd

sys.path.append('/kaggle/working/cogload/')
from install_library import install_library
install_library('pgmpy')

from sklearn.preprocessing import KBinsDiscretizer

class BayesianNetwork:
    def __init__(self):
        pass