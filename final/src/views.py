from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QMainWindow,
    qApp,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QButtonGroup,
    QRadioButton,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, Qt
from numpy import add

from viewmodels import BallastCutterViewModel


class BallastCutterView(QMainWindow):
    def __init__(self):
        super(BallastCutterView, self).__init__()
        self.viewModel = BallastCutterViewModel()
        self.previewRes = (640, 490)
        self.seedRes = (320, 240)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Ballast Cutter")
        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("&File")

        openAction = QAction(QIcon("exit.png"), "Import..", self)
        openAction.setShortcut("Ctrl+O")
        openAction.setStatusTip("Open a file for segmenting.")
        openAction.triggered.connect(self.import_image)
        fileMenu.addAction(openAction)

        saveAction = QAction(QIcon("exit.png"), "Export Labels..", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.setStatusTip("Save file to disk.")
        saveAction.triggered.connect(self.viewModel.export_mask)
        fileMenu.addAction(saveAction)

        exitAction = QAction(QIcon("exit.png"), "&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(qApp.quit)
        fileMenu.addAction(exitAction)

        self.mainWidget = QWidget()
        self.mainLayout = QHBoxLayout()

        self.algoLayout = QVBoxLayout()
        self.combobox = QComboBox()
        self.algos = ["Manual", "Baseline: Graph Cut", "Deep Grab Cut"]
        self.combobox.addItems(self.algos)
        self.combobox.activated[str].connect(self.displayDynamicPanel)
        self.combobox.setCurrentIndex(0)
        self.viewModel.currentAlgo = str(self.combobox.currentText()).lower()

        self.algoLayout.addWidget(self.combobox)

        self.dynamicLayout = QVBoxLayout()
        self.algoLayout.addLayout(self.dynamicLayout)

        clearBtn = QPushButton("Clear All Seeds")
        clearBtn.clicked.connect(self.clear_seeds)
        self.algoLayout.addWidget(clearBtn)

        addAnnotationBtn = QPushButton("Add New Annotation")
        addAnnotationBtn.clicked.connect(self.add_annotation)
        self.algoLayout.addWidget(addAnnotationBtn)

        self.algoLayout.addStretch()
        self.mainLayout.addLayout(self.algoLayout)

        imageLayout = QHBoxLayout()

        previewLayout = QVBoxLayout()
        self.previewLabel = QLabel()
        self.previewLabel.setFixedSize(self.previewRes[0], self.previewRes[1])
        self.previewLabel.mousePressEvent = self.on_previewLabel_mouseDown
        self.previewLabel.mouseMoveEvent = self.on_previewLabel_mouseMove
        self.previewLabel.mouseReleaseEvent = self.on_previewLabel_mouseUp
        previewLayout.addWidget(self.previewLabel)
        previewLayout.addStretch()

        panelLayout = QVBoxLayout()
        self.seedLabel = QLabel()
        self.seedLabel.setFixedSize(self.seedRes[0], self.seedRes[1])
        self.seedLabel.mousePressEvent = self.on_seedLabel_mouseDown
        self.seedLabel.mouseMoveEvent = self.on_seedLabel_mouseMove
        self.seedLabel.mouseReleaseEvent = self.on_seedLabel_mouseRelease

        self.segmentLabel = QLabel()
        self.segmentLabel.setFixedSize(self.seedRes[0], self.seedRes[1])

        intial_pixmap = self.viewModel.get_image_pixmap()
        self.previewLabel.setPixmap(self.scale_pixmap(intial_pixmap, self.previewRes))
        self.seedLabel.setPixmap(self.scale_pixmap(intial_pixmap, self.seedRes))
        self.segmentLabel.setPixmap(self.scale_pixmap(intial_pixmap, self.seedRes))

        panelLayout.addWidget(self.seedLabel)
        panelLayout.addWidget(self.segmentLabel)
        panelLayout.addStretch()

        imageLayout.addLayout(previewLayout)
        imageLayout.addLayout(panelLayout)
        imageLayout.addStretch()
        self.mainLayout.addLayout(imageLayout)

        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)

        self.show()

    def baselinePanel(self):

        self.fgBtn = QRadioButton("Add Foreground Seeds")
        self.fgBtn.clicked.connect(self.viewModel.baseline_fg_mode)
        self.fgBtn.setChecked(True)

        bgBtn = QRadioButton("Add Background Seeds")
        bgBtn.clicked.connect(self.viewModel.baseline_bg_mode)

        select_scene = QButtonGroup()
        select_scene.addButton(self.fgBtn)
        select_scene.addButton(bgBtn)

        self.baseline_advanced_combobox = QComboBox()
        baseline_advanced_operations = ["None", "HSV Color", "Surface Gradient Suppression"]
        self.baseline_advanced_combobox.addItems(baseline_advanced_operations)
        self.baseline_advanced_combobox.activated[str].connect(self.baselineChangeAdvOps)
        self.baseline_advanced_combobox.setCurrentIndex(0)
        self.viewModel.gc_marker.adv_ops = str(self.baseline_advanced_combobox.currentText()).lower()

        self.baseline_neighbor_num_combobox = QComboBox()
        self.baseline_neighbor_num_combobox.addItems(["4", "8"])
        self.baseline_neighbor_num_combobox.activated[str].connect(self.baselineChangeNeighborNum)
        self.baseline_neighbor_num_combobox.setCurrentIndex(0)
        self.viewModel.gc_marker.neighbor_num = 4

        self.baseline_iter_label = QLabel()
        self.baseline_iter_label.setText("Number of Iteration(s): 1")
        self.baseline_iter_num_slider = QSlider(Qt.Horizontal)
        self.baseline_iter_num_slider.setMinimum(1)
        self.baseline_iter_num_slider.setMaximum(10)
        self.baseline_iter_num_slider.setValue(1)
        self.baseline_iter_num_slider.setTickPosition(QSlider.TicksBelow)
        self.baseline_iter_num_slider.setTickInterval(1)
        self.baseline_iter_num_slider.valueChanged[int].connect(self.baselineChangeIterNum)

        runBtn = QPushButton("Run Graph Cut Baseline")
        runBtn.clicked.connect(self.run_algo)

        self.dynamicLayout.addWidget(self.fgBtn)
        self.dynamicLayout.addWidget(bgBtn)
        self.dynamicLayout.addWidget(QLabel("Advanced Configuration:"))
        self.dynamicLayout.addWidget(self.baseline_advanced_combobox)
        self.dynamicLayout.addWidget(QLabel("Number of adjacent pixels:"))
        self.dynamicLayout.addWidget(self.baseline_neighbor_num_combobox)
        self.dynamicLayout.addWidget(self.baseline_iter_label)
        self.dynamicLayout.addWidget(self.baseline_iter_num_slider)
        self.dynamicLayout.addWidget(runBtn)

    def deepGCPanel(self):
        self.deepgc_thres_label = QLabel()
        self.deepgc_thres_label.setText("Threshold: 0.80")
        self.deepgc_thres_slider = QSlider(Qt.Horizontal)
        self.deepgc_thres_slider.setMinimum(1)
        self.deepgc_thres_slider.setMaximum(100)
        self.deepgc_thres_slider.setValue(1)
        self.deepgc_thres_slider.setTickPosition(QSlider.TicksBelow)
        self.deepgc_thres_slider.setTickInterval(1)
        self.deepgc_thres_slider.valueChanged[int].connect(self.deepgcChangeThresh)

        runBtn = QPushButton("Run Deep Grab Cut")
        runBtn.clicked.connect(self.run_algo)
        self.dynamicLayout.addWidget(self.deepgc_thres_label)
        self.dynamicLayout.addWidget(self.deepgc_thres_slider)
        self.dynamicLayout.addWidget(runBtn)

    @staticmethod
    def scale_pixmap(pixmap, resolution, keepAspectRatio=True):
        if keepAspectRatio:
            return pixmap.scaled(resolution[0], resolution[1], Qt.KeepAspectRatio)
        else:
            return pixmap.scaled(resolution[0], resolution[1])

    @pyqtSlot()
    def import_image(self):
        initPixmap = self.viewModel.import_image()
        self.previewLabel.setPixmap(self.scale_pixmap(initPixmap, self.previewRes))
        self.seedLabel.setPixmap(self.scale_pixmap(initPixmap, self.seedRes))
        self.segmentLabel.setPixmap(self.scale_pixmap(initPixmap, self.seedRes))

    @pyqtSlot()
    def on_seedLabel_mouseDown(self, event):
        func = None
        if "baseline" in self.viewModel.currentAlgo:
            func = self.viewModel.baseline_seed_mouse_down
        elif "deep" in self.viewModel.currentAlgo:
            self.viewModel.deep_seed_mouse_down(event, self.seedRes)
        if func:
            self.seedLabel.setPixmap(self.scale_pixmap(func(event, self.seedRes), self.seedRes))

    @pyqtSlot()
    def on_seedLabel_mouseMove(self, event):
        func = None
        if "baseline" in self.viewModel.currentAlgo:
            func = self.viewModel.baseline_seed_mouse_drag
        elif "manual" in self.viewModel.currentAlgo:
            func = self.viewModel.manual_seed_mouse_drag
        elif "deep" in self.viewModel.currentAlgo:
            func = self.viewModel.deep_seed_mouse_drag
        if func:
            self.seedLabel.setPixmap(self.scale_pixmap(func(event, self.seedRes), self.seedRes))

    @pyqtSlot()
    def on_seedLabel_mouseRelease(self, event):
        func = None
        if "manual" in self.viewModel.currentAlgo:
            func = self.viewModel.manual_seed_mouse_release
        elif "deep" in self.viewModel.currentAlgo:
            func = self.viewModel.deep_seed_mouse_release
        if func:
            self.seedLabel.setPixmap(self.scale_pixmap(func(event, self.seedRes), self.seedRes))

    @pyqtSlot()
    def on_previewLabel_mouseDown(self, event):
        self.viewModel.preview_mouse_down(event, self.previewRes)

    @pyqtSlot()
    def on_previewLabel_mouseMove(self, event):
        self.previewLabel.setPixmap(
            self.scale_pixmap(self.viewModel.preview_mouse_move(event, self.previewRes), self.previewRes)
        )

    @pyqtSlot()
    def on_previewLabel_mouseUp(self, event):
        seedPixmap, segPixmap = self.viewModel.preview_mouse_up(event, self.previewRes)
        self.seedLabel.setPixmap(self.scale_pixmap(seedPixmap, self.seedRes))
        self.segmentLabel.setPixmap(self.scale_pixmap(segPixmap, self.seedRes))

    @pyqtSlot()
    def run_algo(self):
        func = None
        if "baseline" in self.viewModel.currentAlgo:
            self.fgBtn.setChecked(True)
            self.viewModel.baseline_fg_mode()
            func = self.viewModel.run_baseline
        elif "deep" in self.viewModel.currentAlgo:
            func = self.viewModel.run_deepgc
        if func:
            self.segmentLabel.setPixmap(self.scale_pixmap(func(), self.seedRes))

    @pyqtSlot()
    def clear_seeds(self):
        if "baseline" in self.viewModel.currentAlgo:
            self.fgBtn.setChecked(True)
            self.viewModel.baseline_fg_mode()
        self.seedLabel.setPixmap(self.scale_pixmap(self.viewModel.clear_seeds(), self.seedRes))

    @pyqtSlot()
    def add_annotation(self):
        func = self.viewModel.add_annotation

        self.previewLabel.setPixmap(self.scale_pixmap(func(), self.previewRes))

        self.clear_seeds()

    @staticmethod
    def clearLayout(layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def displayDynamicPanel(self, text):
        self.clearLayout(self.dynamicLayout)
        self.viewModel.currentAlgo = text.lower()
        if "baseline" in text.lower():
            self.baselinePanel()
        elif "deep" in text.lower():
            self.deepGCPanel()

    def baselineChangeIterNum(self, value):
        self.baseline_iter_label.setText(f"Number of Iteration(s): {value}")
        self.viewModel.gc_marker.iter_num = value

    def baselineChangeAdvOps(self, text):
        self.viewModel.gc_marker.adv_ops = text.lower()

    def baselineChangeNeighborNum(self, text):
        self.viewModel.gc_marker.neighbor_num = int(text)

    def deepgcChangeThresh(self, value):
        thresh = 0.8 + 0.2 * (value - 1) / 100.0
        self.deepgc_thres_label.setText(f"Threshold: {thresh:.02f}")
        self.viewModel.gc_marker.thres = thresh
