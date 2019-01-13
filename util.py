import os
import sys
import datetime
import errno
import glob
import numpy as np
from keras import backend as K


def set_log_dir(model_dir, name, per_epoch=False, val_loss=False, create_dir=True):
    # Directory for training logs
    now = datetime.datetime.now()

    now_str = "{:%Y%m%dT%H%M}".format(now)
    log_dir = os.path.join(model_dir, "{}{}".format(name.lower(), now_str))

    # Create log_dir if not exists
    if not os.path.exists(log_dir):
        if create_dir:
            os.makedirs(log_dir)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), log_dir)

    # Path to save after each epoch. Include epoch and val loss
    # placeholders that gets filled by Keras.
    checkpoint_path = os.path.join(log_dir, "{}_model_*epoch*_*val_loss*.h5".format(name.lower()))

    if val_loss:
        checkpoint_path = checkpoint_path.replace("*val_loss*", "{val_loss:.2f}")
        per_epoch = True
    else:
        checkpoint_path = checkpoint_path.replace("_*val_loss*", "")

    if per_epoch:
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")
    else:
        checkpoint_path = checkpoint_path.replace("_*epoch*", "")

    return log_dir, checkpoint_path


def find_model_file(model_dir, by_val_loss=True):
    # file names are expected in format: <name>_model_<timestamp>_<epoch>_<val_loss>.h5
    # _<epoch> and _<val_loss> are optional
    
    path = os.path.join(model_dir, "*.h5")
    all_model_paths = sorted(glob.glob(path))

    if by_val_loss:
        model_path = all_model_paths[0]
        val_loss = float(sys.maxsize)
        for path in all_model_paths:
            filename = os.path.basename(path)
            file = os.path.splitext(filename)[0]
            file_val_loss = file.split('_')[-1]
            if val_loss > float(file_val_loss):
                val_loss = float(file_val_loss)
                model_path = path

    else:
        model_path = all_model_paths[-1]

    return model_path


# Root Mean Squared Loss Function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


class LRDecay:

    def __init__(self, initial_lrate=1e-3, decay_multiple=0.5, epochs_step=25, patience_multiple=3):
        self.initial_lrate = initial_lrate
        self.decay_multiple = decay_multiple
        self.epochs_step = epochs_step
        self.patience_multiple = patience_multiple
        self.patience = self.epochs_step*self.patience_multiple
        self.history_lr = []
        self.r_epoch = 0

    def linear_decay(self, x):
        return self.decay_multiple - (x/1500)

    def reset(self, r_epoch=0):
        self.r_epoch = r_epoch

    # learning rate schedule
    def step_decay(self, epoch, current_lr):
        lrate = current_lr
    
        if self.r_epoch == 0:
            lrate = self.initial_lrate
    
        elif (1+self.r_epoch) % self.epochs_step == 0:
            lrate = current_lr * self.linear_decay(self.r_epoch)
    
        lrate = np.around(lrate, 8)

        # Use the actual epoch to track history rather then
        # this runs epoch
        self.history_lr.append((epoch, current_lr, lrate))

        self.r_epoch += 1
    
        return lrate
