from tensorflow.keras import callbacks
import math


class CosineAnnealingScheduler(callbacks.LearningRateScheduler):
    def __init__(self, epochs_per_cycle, lr_min, lr_max, verbose=0):
        super(callbacks.LearningRateScheduler, self).__init__()
        self.verbose = verbose
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epochs_per_cycle = epochs_per_cycle

    def schedule(self, epoch, lr):
        return self.lr_min + (self.lr_max - self.lr_min) *\
               (1 + math.cos(math.pi * (epoch % self.epochs_per_cycle) / self.epochs_per_cycle)) / 2