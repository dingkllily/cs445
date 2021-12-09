# Author: Kelin Ding

import numpy as np
import cv2


class ManualMarker:
    def __init__(self, image=None):
        if image is None:
            self.image = np.ones([600, 800, 3], dtype=np.uint8) * 255

        else:
            self.image = image

        self.segment = self.image.copy()
        self.candidate = self.image.copy()
        self.mask = np.zeros_like(self.image, dtype=np.uint8)

        self.start = [0, 0]
        self.end = [0, 0]
        self.current_seq = []
        self.labels = []
        self.line_thickness = 1 + np.min(self.image.shape[:2]) // 400

    def reset_image(self, cvimage):
        self.image = cvimage

        self.segment = self.image.copy()
        self.candidate = self.image.copy()
        self.mask = np.zeros_like(self.image, dtype=np.uint8)

        self.start = [0, 0]
        self.end = [0, 0]
        self.current_seq = []
        self.labels = []
        self.line_thickness = 1 + np.min(self.image.shape[:2]) // 400

    def startAnnotation(self, x, y):
        self.start = (x, y)
        self.current_seq.append((x, y))

    def addSeed(self, x, y):
        self.current_seq.append((x, y))

    def endAnnotation(self):
        self.labels.append(self.current_seq)
        self.current_seq = []

    def clear_seeds(self):
        del self.segment
        del self.candidate
        self.segment = self.image.copy()
        self.candidate = self.image.copy()
        self.current_seq = []

    def draw_seed(self, x, y):
        del self.segment
        del self.candidate
        self.segment = self.image.copy()
        self.mask = np.zeros_like(self.image, dtype=np.uint8)
        # empty
        if len(self.current_seq) == 0:
            self.startAnnotation(x, y)

        else:
            self.addSeed(x, y)
            if len(self.current_seq) > 2:
                countour = np.array(self.current_seq)
                cv2.drawContours(self.segment, [countour], 0, (0, 255, 255), -1)
                cv2.drawContours(self.mask, [countour], 0, (255, 255, 255), -1)

        for (xx, yy) in self.current_seq:
            # draw current seed
            cv2.rectangle(self.segment, (xx - 1, yy - 1), (xx + 1, yy + 1), (0, 0, 255), -1)

        for i in range(1, len(self.current_seq)):
            cv2.line(
                self.segment, self.current_seq[i], self.current_seq[i - 1], (0, 0, 255), thickness=self.line_thickness
            )
        if len(self.current_seq) > 2:
            cv2.line(
                self.segment, self.current_seq[-1], self.current_seq[0], (0, 0, 255), thickness=self.line_thickness
            )

        self.candidate = self.segment.copy()

    def draw_candiate(self, x, y):
        del self.candidate
        self.candidate = self.image.copy()

        if len(self.current_seq) > 1:
            contour = np.array(self.current_seq)
            contour = np.append(contour, [(x, y)], axis=0)
            cv2.drawContours(self.candidate, [contour], 0, (0, 255, 255), -1)

        for (xx, yy) in self.current_seq:
            # draw current seed
            cv2.rectangle(self.candidate, (xx - 1, yy - 1), (xx + 1, yy + 1), (0, 0, 255), -1)

        cv2.rectangle(self.candidate, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)

        for i in range(1, len(self.current_seq)):
            cv2.line(
                self.candidate, self.current_seq[i], self.current_seq[i - 1], (0, 0, 255), thickness=self.line_thickness
            )

        if len(self.current_seq) > 0:
            cv2.line(self.candidate, self.current_seq[-1], (x, y), (0, 0, 255), thickness=self.line_thickness)

        if len(self.current_seq) > 1:
            cv2.line(self.candidate, (x, y), self.current_seq[0], (0, 0, 255), thickness=self.line_thickness)

    def get_seg_overlay(self):
        return cv2.addWeighted(self.image, 0.3, self.segment, 0.6, 0.1)

    def get_candidate_overlay(self):
        return cv2.addWeighted(self.image, 0.3, self.candidate, 0.6, 0.1)
