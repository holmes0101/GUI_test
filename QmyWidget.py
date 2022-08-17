# -*-coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog, QMainWindow

from PyQt5.QtCore import pyqtSlot, QSize, Qt
from PyQt5.QtGui import QIcon, QPixmap, QImage


from Ui_mainwindow import Ui_MainWindow

import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
import cv2

class QmyWidgets(QMainWindow, QWidget):
    def __init__(self, parent=None, args=None, classes=None):
        QMainWindow.__init__(self)
        QWidget.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.args = args
        self.classes = classes

        ## set loadImg button whether is valid
        self.loadImg_btn_flag = True
        self.net = self._load_model(args.model_path)
        self.image_path = ''
        self.ui.loadImg_btn.clicked.connect(self.loadImg_slot)
        self.laest_img_dir = '/'
        self.show_scale = 0.7

    def _scaleImg(self, imgPath):
        oriImg = QImage(imgPath)
        oriSize = oriImg.size()
        resizeWidth = int(oriSize.width() * self.show_scale)
        resizeHeight = int(oriSize.height() * self.show_scale)
        resizedSize = QSize(resizeWidth, resizeHeight)
        resizedImg = QPixmap.fromImage(oriImg.scaled(resizedSize, Qt.IgnoreAspectRatio))
        return resizedImg

    def loadImg_slot(self):
        if self.loadImg_btn_flag:
            #self.loadImg_btn_flag = False
            #self.ui.loadImg_btn.setEnabled(self, False)
            self.image_path = ""

            self.image_path, _ = QFileDialog.getOpenFileName(self, 'open Image file', self.laest_img_dir , "Image files (*.png)")
            print("try:{}".format(self.image_path))

            self.laest_img_dir = '/'.join(self.image_path.split('/')[0:-1])

            self.loadImg_btn_flag = False
            if self.image_path != "":
                '''
                oriImg = QPixmap(self.image_path)
                oriSize = oriImg.size()
                resizeWidth = int(oriSize.width() * self.show_scale)
                resizeHeight = int(oriSize.height() * self.show_scale )
                resizeSize = QSize(resizeHeight, resizeWidth)
                print("oriImg shape:{}".format(resizeWidth))
                print("scale shape:{}".format(resizeSize))
                self.ui.label.setPixmap(QPixmap(self.image_path))
                '''
                resizedImg = self._scaleImg(self.image_path)
                self.ui.label.setPixmap(resizedImg)
                self.ui.label.setScaledContents(True)
                dt_result = self._get_result_from_one_image()
                img = cv2.imread(self.image_path)
                for dt in dt_result:
                    score = round(dt[4], 4)
                    label = dt[5]
                    label = self.classes[int(label)]
                    p1 = (int(dt[0]), int(dt[1]))
                    p2 = (int(dt[0] + dt[2] - 1), int(dt[1] + dt[3] - 1))
                    cv2.rectangle(img, p1, p2, thickness=2, color=(255, 0, 0))
                    cv2.putText(img, str(label + str(score)), p1, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                saveImgDir = './pred.png'
                cv2.imwrite(saveImgDir, img)
                #self.ui.label_2.setPixmap(QPixmap(saveImgDir))
                dt_resizedImg = self._scaleImg(saveImgDir)
                self.ui.label_2.setPixmap(dt_resizedImg)
                self.ui.label_2.setScaledContents(True)
        else:
            self.ui.loadImg_btn.setEnabled(self, False)
        self.loadImg_btn_flag = True

    def _load_model(self, model_path):
        print('Loading model..')
        net = RetinaNet(num_classes=1)
        if torch.cuda.is_available():  ## GPU
            net.load_state_dict(torch.load(model_path)['net'])
        else:  ## cpu
            net.load_state_dict(torch.load(model_path, map_location='cpu')['net'])
        #net.load_state_dict(torch.load(model_path)['net'])
        net.eval()
        print('Loading model done')
        return net

    def _generate_aug_origin(self):
        # Data
        transform = transforms.Compose([
            transforms.ToTensor(),  ## rescale [0, 1] from [H,W,C] to [C, H,W]
            transforms.Normalize(mean=(1, 1, 1), std=(1, 1, 1))
        ])
        return transform

    def _resize_in_test_phase(self, img, size, max_size=1000):
        '''Resize the input PIL image to the given size.

        Args:
          img: (cv2. image) image to be resized.
          size: (tuple or int)
            - if is int, resize the shorter side to the size while maintaining the aspect ratio.
          max_size: (int) when size is int, limit the image longer size to max_size.
                    This is essential to limit the usage of GPU memory.
        Returns:
          img: (cv2. image) resized image.
          scale:
        '''
        h, w = img.shape[:2]
        if isinstance(size, int):
            size_min = min(w, h)
            size_max = max(w, h)
            sw = sh = float(size) / size_min
            if sw * size_max > max_size:
                sw = sh = float(max_size) / size_max
            ow = int(w * sw + 0.5)
            oh = int(h * sh + 0.5)
        else:
            ow, oh = size
            sw = float(ow) / w
            sh = float(oh) / h
        return cv2.resize(img, (int(ow), int(oh))), \
               (sw, sh)

    def _get_result_from_one_image(self):
        '''

        :param net:
        :param image_path:
        :return:   dtboxes: [[xmin, ymin, xmax, ymax, score, label_encode], [], []]
        '''
        ## preproceing before sending Net
        transform = self._generate_aug_origin()

        # print('cur_image:{}'.format(image_path))
        try:
            self.net.cuda(device=self.args.gpu_id)
        except:
            self.net.cpu()

        img = cv2.imread(self.image_path)
        print("img shape:{}".format(img.shape[:2]))
        h, w = img.shape[:2]

        ## with small side resize or with max side resize
        img_resize, scale = self._resize_in_test_phase(img, size=(self.args.input_size, self.args.input_size))

        x = transform(img_resize)
        x = x.unsqueeze(0)  ## add axis  squeeze is delete axis only this anix shape is 1
        with torch.no_grad():
            try:
                x = Variable(x.cuda(device=self.args.gpu_id))
            except:
                x = Variable(x)
            loc_preds, cls_preds = self.net(x)

            loc_preds = loc_preds.data.cpu()  ## shape: [1, w * h, 4] w and h is related with test image size
            cls_preds = cls_preds.data.cpu()  ## shape: [1, w * h, 1]
            encoder = DataEncoder()
            resize_h, resize_w = img_resize.shape[:2]

            boxes, labels, scores = encoder.decode(loc_preds=loc_preds.data.squeeze(0),
                                                   cls_preds=cls_preds.data.squeeze(0),
                                                   input_size=(resize_w, resize_h),
                                                   NMS_THRESH=self.args.nms_thresh,
                                                   CLS_THRESH=self.args.score_thresh)

            boxes = boxes.numpy()  ## [xmin, ymin, xmax, ymax]
            labels = labels.numpy()
            scores = scores.numpy()
        dtboxes = []
        for idx in range(boxes.shape[0]):
            box = boxes[idx]
            box[0] = max(0, box[0] / scale[0])
            box[1] = max(0, box[1] / scale[1])
            box[2] = min(w - 1, box[2] / scale[0])
            box[3] = min(h - 1, box[3] / scale[1])
            box_w = box[2] - box[0] + 1
            box_h = box[3] - box[1] + 1
            if box[0] >= w or box[1] >= h or box_w <= 0 or box_h <= 0:
                continue
            box_list = [box[0], box[1], box_w, box_h, scores[idx], labels[idx]]
            dtboxes.append(box_list)
        return dtboxes



