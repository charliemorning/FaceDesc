# -*- coding: utf-8 -*-

import datetime


class Network(object):
    def __init__(self, model=None):
        self.model = model

    def build_network(self, input_shape, output_shape):
        pass

    def run_generator(self,
                      data_generator,
                      loss,
                      optimizer,
                      metrics=['accuracy'],
                      val_data=None,
                      steps_per_epoch=1,
                      epoch=50,
                      augment=False,
                      callbacks=None):
        if not augment:
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            self.model.summary()
        try:
            self.model.fit_generator(generator=data_generator,
                                     steps_per_epoch=steps_per_epoch,
                                     shuffle=True,
                                     epochs=epoch,
                                     verbose=1,
                                     validation_data=val_data,
                                     callbacks=callbacks)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save("KeyboardInterrupt_" + datestr + ".h5")

    def run(self,
            data,
            loss,
            optimizer,
            metrics=['accuracy'],
            val_data=None,
            batch_size=50,
            epoch=50,
            augment=False,
            callbacks=None):
        if not augment:
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            self.model.summary()
        try:
            self.model.fit(x=data[0], y=data[1],
                           shuffle=True,
                           batch_size=batch_size,
                           epochs=epoch,
                           verbose=1,
                           validation_data=val_data,
                           callbacks=callbacks)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            datestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save("KeyboardInterrupt_" + datestr + ".h5")

    def save(self, path, overwrite=True):
        self.model.save(path, overwrite)

    def save_weights(self, path, overwrite=True):
        self.model.save_weights(path, overwrite)

