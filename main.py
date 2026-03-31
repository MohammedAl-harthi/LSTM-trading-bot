import sys
import os

# Suppress TensorFlow info/warning messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LSTM Futures Bot")


    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
