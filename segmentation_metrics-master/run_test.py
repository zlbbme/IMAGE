import seg_metrics.seg_metrics as sg
labels = [0, 1]

gdth_file = 'segmentation_metrics-master/img/l8_mask.nii.gz'  # ground truth image full path
pred_file = 'segmentation_metrics-master/img/l8_seg.nii.gz'  # prediction image full path
csv_file = 'metrics.csv'

metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                  gdth_path=gdth_file,
                  pred_path=pred_file,
                  csv_file=csv_file)


