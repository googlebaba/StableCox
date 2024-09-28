import logging
import os
import time
from utils import get_expname

# set logger
class Logger():
    def __init__(self, args):
        self.txtname = time.strftime('%Y_%m%d_%H%M%S', time.localtime(time.time()))
        self.expname = get_expname(args)
        self.logpath = os.path.join(args.result_dir, self.expname, self.txtname)
        os.makedirs(os.path.join(args.result_dir, self.expname), exist_ok=True)

        self.logger = logging.getLogger()
        logging.getLogger('matplotlib.font_manager').disabled = True
        self.logger.setLevel(logging.DEBUG) # Here was logging.DEBUG...
        self.fh = logging.FileHandler(self.logpath, mode='a')
        self.fh.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)

    def log_args(self, args):
        for k, v in vars(args).items():
            self.logger.info('%s: %s' % (k, str(v)))
        self.logger.info("")
    
    def debug(self, string):
        self.logger.debug(string)

    def info(self, string):
        self.logger.info(string)
