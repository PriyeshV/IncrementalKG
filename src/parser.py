import argparse
import numpy as np
from datetime import datetime


# TODO # Negative Contraints
class Parser(object):  #
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Dataset settings
        parser.add_argument("--dataset", default='Dummy', help="Dataset to evluate | Check Datasets folder",
                            choices=['Dummy', 'FB20K'])
        parser.add_argument("--n_augments", default=5, type=int)
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)

        # Processing settings
        parser.add_argument("--n_nodes_batch", default=20, type=int)

        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        self.parser = parser

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser