# Author: Kelin Ding
# Reference: https://github.com/NathanZabriskie/GraphCut
import sys

import cv2
import numpy as np
import maxflow


class GraphCutMarker:

    foreground = 1
    background = 0

    seeds = 0
    segmented = 1

    default = 0.5

    def __init__(self, image=None):
        self.image = image
        if image is None:
            self.image = np.ones([600, 800, 3], dtype=np.uint8) * 255

        self.graph = np.zeros_like(self.image)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None

        self.background_seeds = []
        self.foreground_seeds = []
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds
        self.adv_ops = "none"
        self.neighbor_num = 4
        self.iter_num = 1

    def reset_image(self, cvimage):
        self.image = cvimage
        self.graph = np.zeros_like(self.image)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None

        self.background_seeds = []
        self.foreground_seeds = []
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds
        self.neighbor_num = 4

    def load_image(self, filename):
        self.image = cv2.imread(filename)
        self.graph = np.zeros_like(self.image)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None

    def add_seed(self, x, y, type):
        if self.image is None:
            print("Please load an image before adding seeds.")
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1)

    def clear_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)

    def get_overlay(self):
        if self.current_overlay == self.seeds:
            return self.seed_overlay
        else:
            return self.segment_overlay

    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.image, 0.9, self.segment_overlay, 0.4, 0.1)

    def gen_heatmap(self):
        fg_seeds = np.array(self.foreground_seeds)
        cx, cy = np.mean(fg_seeds, axis=0).astype(np.int32).tolist()
        H, W, C = self.image.shape
        dh = np.square(np.arange(H) - cy)
        dw = np.square(np.arange(W) - cx)

        dist = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            dist[i, :] += dw
        for i in range(W):
            dist[:, i] += dh

        dist = np.exp(-dist / 0.0001)
        self.heatmap = dist / dist.max()

    def create_graph(self):
        if len(self.background_seeds) == 0 or len(self.foreground_seeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return

        print("Making graph")
        if "surface gradient" in self.adv_ops:
            self.gen_heatmap()
            self.blurred = cv2.pyrUp(cv2.pyrDown(self.image))
            self.blurred = cv2.resize(self.image, (self.image.shape[1], self.image.shape[0]))
            self.cmp_img = (
                self.image * (1 - self.heatmap[:, :, np.newaxis]) + self.blurred * self.heatmap[:, :, np.newaxis]
            )
        elif "hsv" in self.adv_ops:
            self.hsvimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        for i in range(self.iter_num):
            print(f"[Baseline] Iteration #{i:02d}")
            print("[Baseline] Finding foreground and background averages")
            self.find_averages()

            print("[Baseline] Populating nodes and edges")
            self.populate_graph()

            print("[Baseline] Cutting graph")
            self.cut_graph()

    def find_averages(self):
        self.graph = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.graph.fill(self.default)
        for coordinate in self.background_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 0

        for coordinate in self.foreground_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 1

    def get_gradient(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        if "none" in self.adv_ops:
            return np.sum(np.square(self.image[y1, x1] - self.image[y2, x2]))

        elif "hsv" in self.adv_ops:
            return np.sum(np.square(self.hsvimage[y1, x1] - self.hsvimage[y2, x2]))

        elif "surface gradient" in self.adv_ops:

            return np.sum(np.square(self.cmp_img[y1, x1] - self.cmp_img[y2, x2]))

    def grad_to_weight(self, gradient, mode=1):
        if mode == 0:
            return np.exp(-(gradient ** 2) / 16)

        if mode == 1:
            return 1 / (1 + gradient)

    def populate_graph(self):
        self.nodes = []
        self.edges = []

        # make all s and t connections for the graph
        for (y, x), value in np.ndenumerate(self.graph):
            # this is a background pixel
            if value == 0.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), sys.maxsize, 0))
            # this is a foreground node
            elif value == 1.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, sys.maxsize))
            else:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, 0))

        for (y, x), value in np.ndenumerate(self.graph):
            if y == self.graph.shape[0] - 1 or x == self.graph.shape[1] - 1:
                continue
            my_index = self.get_node_num(x, y, self.image.shape)

            neighbor_index = self.get_node_num(x + 1, y, self.image.shape)
            g = self.grad_to_weight(self.get_gradient((x, y), (x + 1, y)))
            self.edges.append((my_index, neighbor_index, g))

            neighbor_index = self.get_node_num(x, y + 1, self.image.shape)
            g = self.grad_to_weight(self.get_gradient((x, y), (x, y + 1)))
            self.edges.append((my_index, neighbor_index, g))

            if self.neighbor_num == 8 and y > 0:
                neighbor_index = self.get_node_num(x + 1, y - 1, self.image.shape)
                g = self.grad_to_weight(self.get_gradient((x, y), (x + 1, y - 1)))
                self.edges.append((my_index, neighbor_index, g))

                neighbor_index = self.get_node_num(x + 1, y + 1, self.image.shape)
                g = self.grad_to_weight(self.get_gradient((x, y), (x + 1, y + 1)))
                self.edges.append((my_index, neighbor_index, g))

    def cut_graph(self):
        self.segment_overlay = np.zeros_like(self.segment_overlay)
        self.mask = np.zeros_like(self.image, dtype=bool)
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))
        nodelist = g.add_nodes(len(self.nodes))

        for node in self.nodes:
            g.add_tedge(nodelist[node[0]], node[1], node[2])

        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        flow = g.maxflow()

        for index in range(len(self.nodes)):
            if g.get_segment(index) == 1:
                xy = self.get_xy(index, self.image.shape)
                # print(xy[1], xy[0])
                self.segment_overlay[xy[1], xy[0]] = (255, 0, 255)
                self.mask[xy[1], xy[0]] = (True, True, True)

    def swap_overlay(self, overlay_num):
        self.current_overlay = overlay_num

    def save_image(self, filename):
        if self.mask is None:
            print("Please segment the image before saving.")
            return

        to_save = np.zeros_like(self.image)

        np.copyto(to_save, self.image, where=self.mask)
        cv2.imwrite(str(filename), to_save)

    @staticmethod
    def get_node_num(x, y, array_shape):
        return y * array_shape[1] + x

    @staticmethod
    def get_xy(nodenum, array_shape):
        return int(nodenum % array_shape[1]), int(nodenum / array_shape[1])
