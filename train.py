import argparse
import yaml
import tensorflow as tf
from tensorflow import keras

from utils.dataset import DIV2K_Dataset
from utils.model import create_model
from utils.constants import HR_IMG_SIZE, DOWNSAMPLE_MODE


def train(config_fn: str) -> None:
    with open(config_fn, 'r') as stream:
        config = yaml.safe_load(stream)

    train_dataset = DIV2K_Dataset(
        hr_image_folder=config["data_path"],
        batch_size=config["batch_size"],
        set_type="train",
    )
    val_dataset = DIV2K_Dataset(
        hr_image_folder=config["data_path"],
        batch_size=config["val_batch_size"],
        set_type="val",
    )

    model = create_model(d=config["model_d"], s=config["model_s"], m=config["model_m"])
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=config["lr_init"]),
        loss="mean_squared_error",
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=20, min_lr=10e-6, verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=10e-6, 
        patience=40, 
        verbose=0,
        restore_best_weights=True,
    )
    save = keras.callbacks.ModelCheckpoint(
        filepath=config["weights_fn"],
        monitor="loss",
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
    )

    history = model.fit(
        train_dataset,
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        callbacks=[reduce_lr, early_stopping, save], 
        validation_data=val_dataset,
        validation_steps=config["validation_steps"],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    train(config_fn=args.config)