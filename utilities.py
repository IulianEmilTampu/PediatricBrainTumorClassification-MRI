import os
import numpy as np
from matplotlib import pyplot as plt


def plotModelPerformance_v2(
    tr_loss, tr_acc, val_loss, val_acc, save_path, display=False, best_epoch=None
):
    """
    Saves training and validation curves.
    INPUTS
    - tr_loss: training loss history
    - tr_acc: training accuracy history
    - val_loss: validation loss history
    - val_acc: validation accuracy history
    - save_path: path to where to save the model
    """

    fig, ax1 = plt.subplots(figsize=(15, 10))
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "pink",
        "gray",
        "purple",
        "brown",
        "olive",
        "cyan",
        "teal",
    ]
    line_style = [":", "-.", "--", "-"]
    ax1.set_xlabel("Epochs", fontsize=15)
    ax1.set_ylabel("Loss", fontsize=15)
    l1 = ax1.plot(tr_loss, colors[0], ls=line_style[2])
    l2 = ax1.plot(val_loss, colors[1], ls=line_style[3])
    plt.legend(["Training loss", "Validation loss"])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Accuracy and F1-score", fontsize=15)
    ax2.set_ylim(bottom=0, top=1)
    l3 = ax2.plot(tr_acc, colors[2], ls=line_style[2])
    l4 = ax2.plot(val_acc, colors[3], ls=line_style[3])
    if best_epoch:
        l7 = ax2.axvline(x=best_epoch)

    # add legend
    if best_epoch:
        lns = l1 + l2 + l3 + l4 + l7
        labs = [
            "Training loss",
            "Validation loss",
            "Training accuracy",
            "Validation accuracy",
            "Best_model",
        ]
        ax1.legend(lns, labs, loc=7, fontsize=15)
    else:
        lns = l1 + l2 + l3 + l4
        labs = [
            "Training loss",
            "Validation loss",
            "Training accuracy",
            "Validation accuracy",
        ]
        ax1.legend(lns, labs, loc=7, fontsize=15)

    ax1.set_title("Training loss, accuracy and Dice-score trends", fontsize=20)
    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(os.path.join(save_path, "perfomance.pdf"), bbox_inches="tight", dpi=100)
    fig.savefig(os.path.join(save_path, "perfomance.png"), bbox_inches="tight", dpi=100)
    plt.close()

    if display is True:
        plt.show()
    else:
        plt.close()


def get_performance_metrics(true_logits, pred_softmax, average="macro"):
    from sklearn.metrics import (
        average_precision_score,
        recall_score,
        roc_auc_score,
        f1_score,
        accuracy_score,
        matthews_corrcoef,
        confusion_matrix,
    )

    """
    Utility that returns the evaluation metrics as a disctionary.
    THe metrics that are returns are:
    - accuracy
    - f1-score
    - precision and recall
    - auc
    - MCC
    INPUT
    Ytest : np.array
        Array containing the ground truth for the test data
    Ptest_softmax : np.array
        Array containing the softmax output of the model for each
        test sample
    OUTPUT
    metrics_dict : dictionary
    """
    # compute confusion matrix
    cnf_matrix = confusion_matrix(
        np.argmax(true_logits, axis=-1), np.argmax(pred_softmax, axis=-1)
    )

    # get TP, TN, FP, FN
    FP = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
    FN = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
    TP = (np.diag(cnf_matrix)).astype(float)
    TN = (cnf_matrix.sum() - (FP + FN + TP)).astype(float)

    # compute class metrics
    summary_dict = {
        "precision": TN / (FP + TN),
        "recall": TP / (TP + FN),
        "accuracy": (TP + TN) / (TP + TN + FP + FN),
        "f1-score": TP / (TP + 0.5 * (FP + FN)),
        "auc": roc_auc_score(true_logits, pred_softmax, average=None),
    }

    # compute overall metrics
    summary_dict["overall_precision"] = average_precision_score(
        true_logits, pred_softmax, average=average
    )
    summary_dict["overall_recall"] = recall_score(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
        average=average,
    )
    summary_dict["overall_accuracy"] = accuracy_score(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
    )
    summary_dict["overall_f1-score"] = f1_score(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
        average=average,
    )
    summary_dict["overall_auc"] = roc_auc_score(
        true_logits,
        pred_softmax,
        average=average,
        multi_class="ovr",
    )
    summary_dict["matthews_correlation_coefficient"] = matthews_corrcoef(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
    )

    return summary_dict
