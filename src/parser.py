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
        parser.add_argument("--gcnKernel", default='rel_gcn', help="kernel names", choices=['rel_gcn'])

        # Processing settings
        parser.add_argument("--n_nodes_batch", default=20, type=int)
        parser.add_argument("--lr", default=1e-2, help="Learning rate", type=float)
        parser.add_argument("--dropout", default=0.5, help="Dropout", type=float,
                            choices=np.round(np.arange(0, 1, 0.05), 2))
        parser.add_argument("--l2", default=1e-3, help="L2 loss", type=float)
        parser.add_argument("--bias", default=True, type=self.str2bool)

        parser.add_argument("--max_epochs", default=5, help="Max epochs", type=int)
        parser.add_argument("--drop_lr", default=True, help="Drop lr with patience drop", type=self.str2bool)
        parser.add_argument("--pat", default=30, help="Patience", type=int)
        parser.add_argument("--save_after", default=50, help="Save after epochs", type=int)
        parser.add_argument("--val_freq", default=1, help="Validation frequency", type=int)
        parser.add_argument("--summaries", default=True, help="Save summaries after each epoch", type=self.str2bool)

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