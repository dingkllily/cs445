import random

import cv2
import numpy as np
from collections import defaultdict

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap

from algos.GraphCutMarker import GraphCutMarker
from algos.ManualMarker import ManualMarker


class BallastCutterViewModel:
    def __init__(self) -> None:

        # algos
        self.gc_marker = GraphCutMarker()
        self.manual_marker = ManualMarker()

        self.seed_num = self.gc_marker.foreground

        self.image = self.gc_marker.image.copy()
        self.previewImg = self.image
        self.partitions = np.zeros_like(self.gc_marker.image[:, :, 0], dtype=np.int32)
        self.colorSeg = self.image.copy()
        self.labels = defaultdict(list)
        self.selectStart = [0, 0]
        self.selectEnd = [-1, -1]

        self.currentAlgo = "Manual"

    @staticmethod
    def get_qimage(cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix
        rgb = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def set_img(self, fn):
        self.image = cv2.imread(fn)
        self.previewImg = self.image.copy()

        if "manual" in self.currentAlgo:
            self.manual_marker.reset_image(self.image.copy())
        elif "baseline" in self.currentAlgo:
            self.gc_marker.load_image(fn)

        self.partitions = np.zeros_like(self.image[:, :, 0], dtype=np.int32)
        self.labels = defaultdict(list)
        self.selectStart = [0, 0]
        self.selectEnd = self.image.shape[:2]
        self.colorSeg = self.image.copy()

    @staticmethod
    def remap_event(event, image, label_shape):
        ih, iw = image.shape[:2]
        x, y = event.x(), event.y()
        lw, lh = label_shape
        img_aspectRatio = iw / ih
        label_aspectRatio = lw / lh

        if img_aspectRatio > label_aspectRatio:
            offsetW = 0
            scale = lw / iw
            offsetH = (lh - ih * scale) // 2
        else:
            offsetH = 0
            scale = lh / ih
            offsetW = (lw - iw * scale) // 2

        ix = (x - 0) / scale
        iy = (y - offsetH) / scale

        ix = min(image.shape[1] - 1, max(0, ix))
        iy = min(image.shape[0] - 1, max(0, iy))

        return int(ix), int(iy)

    def baseline_fg_mode(self):
        self.seed_num = self.gc_marker.foreground

    def baseline_bg_mode(self):
        self.seed_num = self.gc_marker.background

    def baseline_on_clear(self):
        self.gc_marker.clear_seeds()

        return QPixmap.fromImage(self.get_qimage(self.gc_marker.get_image_with_overlay(self.gc_marker.seeds)))

    def baseline_on_segment(self):
        self.gc_marker.create_graph()

        return QPixmap.fromImage(self.get_qimage(self.gc_marker.get_image_with_overlay(self.gc_marker.segmented)))

    def import_image(self):
        f = QFileDialog.getOpenFileName()
        if f is not None and f != "":
            input_img = cv2.imread(str(f[0]))
            if input_img is not None:
                del input_img
                self.set_img(str(f[0]))

        initPixmap = QPixmap.fromImage(self.get_qimage(self.image))

        return initPixmap

    def baseline_seed_mouse_down(self, event, labelRes):
        ix, iy = self.remap_event(event, self.gc_marker.image, labelRes)
        self.gc_marker.add_seed(ix, iy, self.seed_num)

        return QPixmap.fromImage(self.get_qimage(self.gc_marker.get_image_with_overlay(self.gc_marker.seeds)))

    def baseline_seed_mouse_drag(self, event, labelRes):
        ix, iy = self.remap_event(event, self.gc_marker.image, labelRes)
        self.gc_marker.add_seed(ix, iy, self.seed_num)

        return QPixmap.fromImage(self.get_qimage(self.gc_marker.get_image_with_overlay(self.gc_marker.seeds)))

    def manual_seed_mouse_drag(self, event, labelRes):
        ix, iy = self.remap_event(event, self.manual_marker.image, labelRes)
        self.manual_marker.draw_candiate(ix, iy)

        return QPixmap.fromImage(self.get_qimage(self.manual_marker.get_candidate_overlay()))

    def manual_seed_mouse_release(self, event, labelRes):
        ix, iy = self.remap_event(event, self.manual_marker.image, labelRes)
        self.manual_marker.draw_seed(ix, iy)

        return QPixmap.fromImage(self.get_qimage(self.manual_marker.get_seg_overlay()))

    def preview_mouse_down(self, event, labelRes):
        ix, iy = self.remap_event(event, self.image, labelRes)
        self.selectStart = [iy, ix]

    def preview_mouse_move(self, event, labelRes):
        ix, iy = self.remap_event(event, self.image, labelRes)
        starty = min(self.selectStart[0], iy)
        startx = min(self.selectStart[1], ix)
        endy = max(self.selectStart[0], iy)
        endx = max(self.selectStart[1], ix)
        del self.previewImg
        self.previewImg = self.image.copy()
        cv2.rectangle(self.previewImg, (startx, starty), (endx, endy), (255, 255, 255), -1)
        weighted = cv2.addWeighted(self.colorSeg, 0.3, self.previewImg, 0.6, 0.1)

        return QPixmap.fromImage(self.get_qimage(weighted))

    def preview_mouse_up(self, event, labelRes):
        ix, iy = self.remap_event(event, self.image, labelRes)
        starty = min(self.selectStart[0], iy)
        startx = min(self.selectStart[1], ix)
        endy = max(self.selectStart[0], iy)
        endx = max(self.selectStart[1], ix)

        area = (endy - starty) * (endx - startx)
        if area >= 100:
            self.selectStart = [starty, startx]
            self.selectEnd = [endy, endx]
        else:
            self.selectStart = [0, 0]
            self.selectEnd = self.image.shape[:2]
            starty, startx = self.selectStart
            endy, endx = self.selectEnd

        if "baseline" in self.currentAlgo:
            self.gc_marker.reset_image(self.image[starty : endy + 1, startx : endx + 1].copy())
            seedPixmap = QPixmap.fromImage(self.get_qimage(self.gc_marker.get_image_with_overlay(self.gc_marker.seeds)))
            segPixmap = QPixmap.fromImage(
                self.get_qimage(self.gc_marker.get_image_with_overlay(self.gc_marker.segmented))
            )
        elif "manual" in self.currentAlgo:
            self.manual_marker.reset_image(self.image[starty : endy + 1, startx : endx + 1])
            seedPixmap = QPixmap.fromImage(self.get_qimage(self.manual_marker.get_seg_overlay()))
            segPixmap = QPixmap.fromImage(self.get_qimage(self.manual_marker.get_seg_overlay()))

        return seedPixmap, segPixmap

    def get_image_pixmap(self):
        return QPixmap.fromImage(self.get_qimage(self.image))

    def run_baseline(self):
        self.gc_marker.create_graph()
        return QPixmap.fromImage(self.get_qimage(self.gc_marker.get_image_with_overlay(self.gc_marker.segmented)))

    def clear_seeds(self):
        seed_img = None
        if "baseline" in self.currentAlgo:
            self.gc_marker.clear_seeds()
            seed_img = self.gc_marker.get_image_with_overlay(self.gc_marker.seeds)
        elif "manual" in self.currentAlgo:
            self.manual_marker.clear_seeds()
            seed_img = self.manual_marker.get_seg_overlay()
        return QPixmap.fromImage(self.get_qimage(seed_img))

    @staticmethod
    def find_contour_of_mask(mask):
        return [(0, 0)]

    def add_annotation(self):
        starty, startx = self.selectStart
        endy, endx = self.selectEnd
        mask = np.zeros([endy - starty + 1, endx - startx + 1], dtype=np.int32)
        if "baseline" in self.currentAlgo:
            mask = self.gc_marker.mask[:, :, 0].astype(np.bool8)

        elif "manual" in self.currentAlgo:
            mask = self.manual_marker.mask[:, :, 0].astype(np.bool8)

        annotation_id = len(list(self.labels.keys()))
        self.partitions[starty : endy + 1, startx : endx + 1][mask] = annotation_id + 1
        self.labels[annotation_id] = self.find_contour_of_mask(mask)

        # randomRGB = int(random.random() * 1e10)
        # r = (randomRGB >> 0) & 255
        # g = (randomRGB >> 8) & 255
        # b = (randomRGB >> 16) & 255
        r, g, b = 255, 255, 0

        self.colorSeg[starty : endy + 1, startx : endx + 1][mask] = (b, g, r)
        weighted = cv2.addWeighted(self.image, 0.3, self.colorSeg, 0.6, 0.1)

        return QPixmap.fromImage(self.get_qimage(weighted))
