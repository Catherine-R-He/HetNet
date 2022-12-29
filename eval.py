from misc import *
import os
import numpy as np

pred_dir = "/home/crh/AAAI-Mirror-Code/map-PMD/PMD/"
gt_dir = "/home/crh/MirrorDataset/PMD/test/mask/"

print(pred_dir)
# for sub_dir in os.listdir(pred_dir):
#     if 0 == len(os.listdir(os.path.join(pred_dir, sub_dir))):
#         continue
iou_l = []
acc_l = []
mae_l = []
ber_l = []
f_measure_p_l = []
f_measure_r_l = []
precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
for name in os.listdir(os.path.join(pred_dir)):
    #print(name)
    gt = get_gt_mask(name, gt_dir)
    normalized_pred = get_normalized_predict_mask(name, pred_dir)
    binary_pred = get_binary_predict_mask(name, pred_dir)

    if normalized_pred.ndim == 3:
        normalized_pred = normalized_pred[:, :, 0]
    if binary_pred.ndim == 3:
        binary_pred = binary_pred[:, :, 0]

    acc_l.append(accuracy_mirror(binary_pred, gt))
    iou_l.append(compute_iou(binary_pred, gt))
    mae_l.append(compute_mae(normalized_pred, gt))
    ber_l.append(compute_ber(binary_pred, gt))

    pred = (255 * normalized_pred).astype(np.uint8)
    gt = (255 * gt).astype(np.uint8)
    p, r = cal_precision_recall(pred, gt)
    for idx, data in enumerate(zip(p, r)):
        p, r = data
        precision_record[idx].update(p)
        recall_record[idx].update(r)
print('%s:  mae: %4f, ber: %4f, acc: %4f, iou: %4f, f_measure: %4f' %
        (pred_dir, np.mean(mae_l), np.mean(ber_l), np.mean(acc_l), np.mean(iou_l),
        cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])))
