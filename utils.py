import numpy as np
from sklearn.metrics import roc_curve


def get_metric_and_best_threshold_from_roc_curve(true_label, pred_proba, nr=0.2, tpc=9000, th=0.5):
    fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
    #
    # precision, recall, thresholds = precision_recall_curve(true_label, pred_proba)
    num_pos_class, num_neg_class = (true_label == 1).sum(), (true_label == 0).sum()
    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    fp = fpr * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)
    noise_rate = fp / (tp + fp + 1e-7)

    d = (1 - tpr) ** 2 + fpr ** 2
    # my_th_i = np.abs((thresholds - th)).argmin()
    # print(fp.shape, tp.shape, thresholds.shape, (thresholds - th).shape)
    # print("{:.2f} {:.2f} {:.2f}".format(thresholds[my_th_i], tp[my_th_i], fp[my_th_i]))
    th_i1, th_i2, th_i3 = np.abs((noise_rate - nr)).argmin(), np.abs(tp - tpc).argmin(), np.argmin(d)
    return thresholds[th_i1], tp[th_i1] + fp[th_i1], thresholds[th_i2], tp[th_i2] + fp[th_i2], thresholds[th_i3], tp[
        th_i3] + fp[th_i3]
