import tensorflow as tf


# def MCC_Loss(y_true, y_pred):
#     """
#     Calculates the proposed Matthews Correlation Coefficient-based loss.
#     Args:
#         y_true (tf.Tensor): 1-hot encoded predictions
#         y_pred (tf.Tensor): 1-hot encoded ground truth

#     Output:

#         MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
#         where TP, TN, FP, and FN are elements in the confusion matrix.

#     code from https://arxiv.org/pdf/2010.13454.pdf
#     """

#     tp = tf.sum(tf.mul(y_pred, y_true))
#     tn = tf.sum(tf.mul((1 - y_pred), (1 - y_true)))
#     fp = tf.sum(tf.mul(y_pred, (1 - y_true)))
#     fn = tf.sum(tf.mul((1 - y_pred), y_true))

#     numerator = tf.mul(tp, tn) - tf.mul(fp, fn)
#     denominator = tf.sqrt(
#         tf.add(tp, 1, fp) * tf.add(tp, 1, fn) * tf.add(tn, 1, fp) * tf.add(tn, 1, fn)
#     )

#     # Adding 1 to the denominator to avoid divide-by-zero errors.
#     mcc = tf.div(numerator.sum(), denominator.sum() + 1.0)

#     return 1 - mcc


class MCC_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(MCC_Loss, self).__init__()

    def call(self, y_true, y_pred):
        tp = tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=-1)
        tn = tf.math.reduce_sum(tf.math.multiply((1 - y_pred), (1 - y_true)), axis=-1)
        fp = tf.math.reduce_sum(tf.math.multiply(y_pred, (1 - y_true)), axis=-1)
        fn = tf.math.reduce_sum(tf.math.multiply((1 - y_pred), y_true), axis=-1)

        numerator = tf.math.multiply(tp, tn) - tf.math.multiply(fp, fn)
        denominator = tf.math.sqrt(
            tf.math.add(tp, fp)
            * tf.math.add(tp, fn)
            * tf.math.add(tn, fp)
            * tf.math.add(tn, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = tf.math.divide(numerator, denominator + 1.0)

        return 1 - mcc


class MCC_and_CCE_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(MCC_and_CCE_Loss, self).__init__()

    def call(self, y_true, y_pred):
        tp = tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=-1)
        tn = tf.math.reduce_sum(tf.math.multiply((1 - y_pred), (1 - y_true)), axis=-1)
        fp = tf.math.reduce_sum(tf.math.multiply(y_pred, (1 - y_true)), axis=-1)
        fn = tf.math.reduce_sum(tf.math.multiply((1 - y_pred), y_true), axis=-1)

        numerator = tf.math.multiply(tp, tn) - tf.math.multiply(fp, fn)
        denominator = tf.math.sqrt(
            tf.math.add(tp, fp)
            * tf.math.add(tp, fn)
            * tf.math.add(tn, fp)
            * tf.math.add(tn, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = tf.math.divide(numerator, denominator + 0.0000006)

        cce = tf.keras.losses.CategoricalCrossentropy()

        return (1 - mcc) + cce(y_true, y_pred)
