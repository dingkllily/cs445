import cv2
import numpy as np
from PyQt5.QtWidgets import QImage


class MyImage(QImage):
    def __init__(self):
        self.cvImage = np.zeros((100, 100, 3))

        super(MyImage, self).__init__(self.cvImage.data, 100, 100, 300, QImage.Format_RGB888)

    def set_image(self, image):

        height, width, bytes_per_pix = image.shape
        bytes_per_line = width * bytes_per_pix
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, self.cvImage)

        super(MyImage, self).__init__(self.cvImage.data, width, height, bytes_per_line, QImage.Format_RGB888)
