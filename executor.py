import numpy as np
import dataloader
from info import ModelParameters as Params
import model as cool_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger


class Executor:
    def train(self):
        model = cool_model.create_model()
        model.compile(optimizer=Adam(Params.learn_rate), loss=Params.losses, metrics=Params.metrics)
        print(model.summary())

        # callbacks
        logger = CSVLogger(f'out/logs/{Params.model_name}_logs.csv')
        checkpointer = ModelCheckpoint(f'out/checkpoints/{Params.model_name}_{{epoch:02d}}.hdf5', verbose=1,
                                       save_best_only=True)

        # dataloaders
        train_loader = dataloader.Dataloader(dataloader.Dataloader.mode_train, split_size=50000)
        validation_loader = dataloader.Dataloader(dataloader.Dataloader.mode_validation, split_size=int(50000 * .42))

        num_train_batches = train_loader.__len__()
        num_valid_batches = int(num_train_batches * 0.42)

        model.fit_generator(generator=train_loader, validation_data=validation_loader, callbacks=[checkpointer, logger],
                            steps_per_epoch=num_train_batches, epochs=100, validation_steps=num_valid_batches,shuffle=False)


if __name__ == "__main__":
    executor = Executor()
    executor.train()
