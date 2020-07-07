from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch


class Logger(object):
    def __init__(self, opt, train_writer, val_writer):
        self.train_writer = train_writer
        self.val_writer = val_writer
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        # write args in opt.txt
        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))
        f = os.path.join(opt.save_dir, 'opt.txt')
        with open(f, 'wt') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        self.log = open(opt.log_dir + '/log.txt', 'a+')

        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step, type):
        """Log a scalar variable."""
        if type == 'train':
            self.train_writer.add_scalar(tag, value, step)
            self.train_writer.flush()
        else:
            self.val_writer.add_scalar(tag, value, step)
            self.val_writer.flush()
