import sys
from PyQt5.QtWidgets import QApplication


from views import BallastCutterView


def main():
    app = QApplication(sys.argv)
    window = BallastCutterView()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
