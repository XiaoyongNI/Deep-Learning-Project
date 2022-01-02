base_dir_detections_fd = '../dataset/kitti/base_dir_detections_fd' # Directory that contains detections by fine level detector
base_dir_detections_cd = '../dataset/kitti/base_dir_detections_cd' # Directory that contains detections by coarse level detector
base_dir_groundtruth = '../dataset/kitti/base_dir_groundtruth' # Directory that contains ground truth bounding boxes
base_dir_metric_fd = '../dataset/kitti/' # Directory that contains AP or AR values by the fine detector
base_dir_metric_cd = '../dataset/kitti/' # Directory that contains AP or AR values by the coarse detector

num_actions = 16 # Hyperparameter, should be equal to num_windows * num_windows
num_windows = 4 # Number of windows in one dimension
img_size_fd = 880 # Image size used to train the fine level detector
img_size_cd = 220 # Image size used to train the coarse level detector

iou_threshold = 0.75 