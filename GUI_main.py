

import argparse

from QmyWidget import QmyWidgets as QmyW
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Detection')
    parser.add_argument('--model_path', default='./best_model/ckpt_last_epoch.pth', type=str, help='learning rate')
    parser.add_argument('--nms_thresh', default=0.45, type=float, help='nms thresh')
    parser.add_argument('--score_thresh', default=0.3, type=float, help='score thresh')
    parser.add_argument('--input_size', default=416, type=int, help='test input size')
    parser.add_argument('--test_file', '-t', default='JPEGImages/0000716.png',
                         type=str, help='test file or test image')
    parser.add_argument('--test_root', default='./data',type=str, help='test data root')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU index')
    parser.add_argument('--result_dir', default='./detection_result', type=str, help='save result')

    ### just for calculate F1 score in Aug Search
    parser.add_argument('--gt_dt_iou_thresh', default=0.6, type=float, help='gt dt iou thresh to TP')
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    ## get parameter
    global args
    args = parse_args()
    ## get test image or test image list(using darknet test.txt)

    classes = ['missile']
    app = QApplication(sys.argv)
    icon = QIcon("./icon/icon.ico")
    app.setWindowIcon(icon)
    mainUI = QmyW(args=args, classes=classes)
    mainUI.show()
    sys.exit(app.exec_())
