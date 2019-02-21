from keras.callbacks import Callback


class ModelCheckpoint(Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        self.val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.val_loss:
            self.val_loss = logs.get('val_loss')
            self.model.save_weights(self.filepath)
