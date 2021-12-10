# Author: Tianhui Cai
# Reference: https://github.com/NathanZabriskie/GraphCut
import sys

import cv2
import numpy as np
import maxflow
class ReducedMarker:
    def __init__(self, image = None):
        self.image = image
        if image is None:
            self.image = np.ones([600, 800, 3], dtype=np.uint8) * 255
        self.graph = np.zeros_like(self.image)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None
        self.point_input = []
        self.background_seeds = []
        self.foreground_seeds = []
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds
        self.adv_ops = "none"
        self.neighbor_num = 4
        self.iter_num = 1
        self.th_inner = 12
        self.df = 10
        self.innerbox = []
        self.outerbox = []

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

    def add_point(self,x,y):
        if self.image is None:
            print("Please load an image before input the point.")
        self.point_input.clear()
        self.point_input.append((x, y));

    def getinnerbox(self,x,y,th_inner,df):
        # points: (x,y) of the user chosen point
        # image: [H*W*3]
        # th_inner: threshold of the inner box, ideally between 50-80
        # df: int
        # initial inner box
        H, W, _ = self.image.shape
        x0 = x - 2
        x1 = x + 2
        y0 = y - 2
        y1 = y + 2
        up = down = left = right = True
        while (up or down or left or right):
            x0_temp = x0 - df
            x1_temp = x1 + df
            y0_temp = y0 - df
            y1_temp = y1 + df
            if (x0_temp < 0):
                x0_temp = 0
                right = False
            if (x1_temp >= W):
                x1_temp = W - 1
                left = False
            if (y0_temp < 0):
                y0_temp = 0
                up = False
            if (y1_temp >= H):
                y1_temp = H - 1
                down = False

            # up
            E = (np.sum((self.image[y0_temp, x0:x1, :] - self.image[y0, x0:x1, :]) ** 2) ** (0.5)) / abs(x0 - x1)
            if (E <= th_inner):
                y0 = y0_temp
            else:
                up = False
            # left
            E = (np.sum((self.image[y0:y1, x0_temp, :] - self.image[y0:y1, x0, :]) ** 2) ** (0.5)) / abs(y0 - y1)
            if (E <= th_inner):
                x0 = x0_temp
            else:
                left = False
            # down
            E = (np.sum((self.image[y1_temp, x0:x1, :] - self.image[y1, x0:x1, :]) ** 2) ** (0.5)) / abs(x0 - x1)
            if (E <= th_inner):
                y1 = y1_temp
            else:
                down = False
            # right
            E = (np.sum((self.image[y0:y1, x1_temp, :] - self.image[y0:y1, x1, :]) ** 2) ** (0.5)) / abs(y0 - y1)
            if (E <= th_inner):
                x1 = x1_temp
            else:
                right = False
        x0 = x0 + 10
        x1 = x1 - 10
        y0 = y0 + 10
        y1 = y1 - 10
        self.innerbox = [x0,x1,y0,y1]

    def getouterbox(self,th_outer,df):
        H, W, _ = image.shape
        x0 = self.innerbox[0] - 2
        x1 = self.innerbox[1] + 2
        y0 = self.innerbox[2] - 2
        y1 = self.innerbox[3] + 2
        # up
        E = 0
        y0_out = y0
        while (y0_out - df >= 0):
            y0_temp = y0_out - df
            E = (np.sum((self.image[y0_temp, x0:x1, :] - self.image[y0_out, x0:x1, :]) ** 2) ** (0.5)) / abs(x0 - x1)
            y0_out = y0_temp
            if (E > th_outer):
                break
        # left
        x0_out = x0
        while (x0_out - df >= 0):
            x0_temp = x0_out - df
            E = (np.sum((self.image[y0:y1, x0_temp, :] - self.image[y0:y1, x0_out, :]) ** 2) ** (0.5)) / abs(y0 - y1)
            x0_out = x0_temp
            if (E > th_outer):
                break
        # down
        y1_out = y1
        while ((y1_out + df) < H):
            y1_temp = y1_out + df
            E = (np.sum((self.image[y1_temp, x0:x1, :] - self.image[y1_out, x0:x1, :]) ** 2) ** (0.5)) / abs(x0 - x1)
            y1_out = y1_temp
            if (E > th_outer):
                break
        # right
        x1_out = x1
        while ((x1_out + df) < W):
            x1_temp = x1_out + df
            E = (np.sum((self.image[y0:y1, x1_temp, :] - self.image[y0:y1, x1_out, :]) ** 2) ** (0.5)) / abs(y0 - y1)
            x1_out = x1_temp
            if (E > th_outer):
                break
        x0_out = x0_out - 20
        x1_out = x1_out + 20
        y0_out = y0_out - 20
        y1_out = y1_out + 20
        self.outerbox = [x0_out,x1_out,y0_out,y1_out]

    def add_seed(self):
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


    def remove_point(self, x, y):
        self.point_input.clear();
        self.background_seeds.clear();
        self.foreground_seeds.clear();
        self.seed_overlay = np.zeros_like(self.seed_overlay)
        self.outerbox.clear();
        self.innerbox.clear();


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
