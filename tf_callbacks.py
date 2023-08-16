import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, print_every_n_epoch: int = 5):
        self.save_path = save_path
        self.print_every_n_epoch = print_every_n_epoch
        self.history = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": [],
            "learning_rate": [],
            "MatthewsCorrelationCoefficient": [],
            "val_MatthewsCorrelationCoefficient": [],
        }

    def get_epoch_learning_rate(self, epoch):
        try:
            learning_rate_scheduler = tf.keras.backend.eval(self.model.optimizer.lr)
            decay = (
                1
                - (epoch / float(learning_rate_scheduler.decay_steps))
                ** learning_rate_scheduler.power
            )
            alpha = learning_rate_scheduler.initial_learning_rate * decay
            return alpha
        except:
            return None

    def on_epoch_end(self, epoch, logs=None):
        # print([print(key, value) for key, value in self.history.items()])
        # print([print(key, value) for key, value in logs.items()])
        # save variables in self.history
        [
            self.history[key].extend([logs[key]])
            for key in ["accuracy", "val_accuracy", "loss", "val_loss"]
        ]
        # try to add mcc if is avaliable
        try:
            [
                self.history[key].extend([logs[key]])
                for key in [
                    "MatthewsCorrelationCoefficient",
                    "val_MatthewsCorrelationCoefficient",
                ]
            ]
        except:
            pass

        # self.history["learning_rate"].extend([tf.keras.backend.eval(self.model.optimizer.lr)])
        self.history["learning_rate"].extend([self.get_epoch_learning_rate(epoch)])
        # save graphs
        if epoch % self.print_every_n_epoch == 0:
            print(" Saving graphs...")
            x_ = range(epoch + 1)

            fig, ax = plt.subplots(
                figsize=(20, 15),
                nrows=3 if self.history["MatthewsCorrelationCoefficient"] else 2,
                ncols=1,
            )
            # print training loss
            ax[0].plot(self.history["loss"], label="Training loss")
            ax[0].plot(self.history["val_loss"], label="Validation loss")
            ax[0].set_title(f"Training and Validation loss curves")
            ax[0].legend()
            # print training accuracy
            ax[1].plot(self.history["accuracy"], label="Training accuracy")
            ax[1].plot(self.history["val_accuracy"], label="Validation accuracy")
            ax[1].set_title(f"Training and Validation accuracy curves")
            ax[1].legend()

            # print training MCC
            if self.history["MatthewsCorrelationCoefficient"]:
                ax[2].plot(
                    self.history["MatthewsCorrelationCoefficient"], label="Training MCC"
                )
                ax[2].plot(
                    self.history["val_MatthewsCorrelationCoefficient"],
                    label="Validation MCC",
                )
                ax[2].set_title("Training and Validation MCC curves")
                ax[2].legend()

            plt.savefig(
                os.path.join(self.save_path, f"training_curves_e_{epoch+1}.png")
            )
            plt.close(fig)

            if self.history["learning_rate"][-1]:
                # save also learning rate
                plt.figure(figsize=(8, 8))
                plt.plot(
                    x_,
                    self.history["learning_rate"],
                    label="Learning_rate",
                )
                plt.legend(loc="upper right")
                plt.title("Learning rate")
                plt.savefig(os.path.join(self.save_path, "learning_rate.png"))
                plt.close()


class SaveBestModelWeights(tf.keras.callbacks.Callback):
    def __init__(self, save_path, monitor="validation_loss", mode="min"):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        if mode == "max":
            self.best = float("-inf")
        else:
            self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.monitor]

        if self.mode == "max":
            if metric_value > self.best:
                print(
                    f"Saving model weight Callback. Monitor {self.monitor}, mode {self.mode}"
                )
                print(f"Current best: {self.best}")
                print(f"Current value: {metric_value}")
                self.best = metric_value
                # self.best_weights = self.model.get_weights()
                # save model meights
                print(f"Saving model weights at: {self.save_path}")
                # self.model.save_weights(self.save_path, overwrite=True)
                self.model.save(
                    os.path.join(self.save_path, f"best_model"),
                    save_format="h5",
                    include_optimizer=False,
                )
        else:
            if metric_value < self.best:
                print(
                    f"Saving model weight Callback. Monitor {self.monitor}, mode {self.mode}"
                )
                print(f"Current best: {self.best}")
                print(f"Current value: {metric_value}")
                self.best = metric_value
                self.best_weights = self.model.get_weights()
                # save model meights
                print(f"Saving model weights at: {self.save_path}")
                self.model.save(
                    os.path.join(self.save_path, f"best_model"),
                    save_format="h5",
                    include_optimizer=False,
                )

class LRFind(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, n_rounds):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_up = (max_lr / min_lr) ** (1 / n_rounds)
        self.lrs = []
        self.losses = []

    def on_train_begin(self, logs=None):
        self.weights = self.model.get_weights()
        self.model.optimizer.lr = self.min_lr

    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs["loss"])
        self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)


# %%

# importlib.reload(tf_callbacks)

# model = models.SimpleDetectionModel_TF(
#             num_classes=args_dict["NBR_CLASSES"],
#             input_shape=input_shape,
#             image_normalization_stats=norm_stats[0],
#             scale_image=args_dict["DATA_SCALE"],
#             data_augmentation=args_dict["DATA_AUGMENTATION"],
#             kernel_size=(3, 3),
#             pool_size=(2, 2),
#             use_age=args_dict["USE_AGE"],
#             age_normalization_stats=norm_stats[2],
#             use_age_thr_tabular_network=False,
#             use_pretrained=False,
#             pretrained_model_path=None,
#             freeze_weights=None,
#         )

# model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss=tf.keras.losses.CategoricalCrossentropy(),
#         metrics=[
#             "accuracy"])

# EPOCHS = 5
# BATCH_SIZE = 32
# lr_finder_steps = train_steps
# lr_find = tf_callbacks.LRFind(1e-4, 1e-1, lr_finder_steps)

# model.fit(
#     train_gen,
#     steps_per_epoch=train_steps,
#     epochs=5,
#     callbacks=[lr_find]
# )

# plt.plot(lr_find.lrs, lr_find.losses)
# plt.xscale('log')
# plt.show()
