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
        # self.history["learning_rate"].extend([tf.keras.backend.eval(self.model.optimizer.lr)])
        self.history["learning_rate"].extend([self.get_epoch_learning_rate(epoch)])
        # save graphs
        if epoch % self.print_every_n_epoch == 0:
            print(" Saving graphs...")
            x_ = range(epoch + 1)
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(
                x_,
                self.history["accuracy"],
                label="Training Accuracy",
            )
            plt.plot(
                x_,
                self.history["val_accuracy"],
                label="Validation Accuracy",
            )
            plt.legend(loc="lower right")
            plt.title("Training and Validation Accuracy")
            plt.subplot(2, 1, 2)
            plt.plot(
                x_,
                self.history["loss"],
                label="Training Loss",
            )
            plt.plot(
                x_,
                self.history["val_loss"],
                label="Validation Loss",
            )
            plt.legend(loc="upper right")
            plt.title("Training and Validation Loss")
            plt.savefig(os.path.join(self.save_path, "acc_loss.png"))
            plt.close()

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
        print(f"Saving model weight callBack. Monitor {self.monitor}, mode {self.mode}")
        print(f"Current best: {self.best}")
        print(f"Current value: {metric_value}")

        if self.mode == "max":
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()
                # save model meights
                print(f"Saving model weights at: {self.save_path}")
                self.model.save_weights(self.save_path, overwrite=True)
        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()
                # save model meights
                print(f"Saving model weights at: {self.save_path}")
                self.model.save_weights(self.save_path, overwrite=True)


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

    def get_config(self):

        config = {
            "learning_rate_base": self.learning_rate_base,
            "total_steps": self.total_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
            "pi": self.pi,
        }
        return config
