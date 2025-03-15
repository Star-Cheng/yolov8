import sys
import cv2
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import ui_img.detect_images_rc


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setWindowTitle("基于YOLOv8的检测演示软件V1.0")
        self.resize(1500, 1000)
        self.setStyleSheet("QWidget#centralwidget{background-image: url(:/detect_background/detect.JPG);}")
        self.centralwidget = QWidget()
        self.centralwidget.setObjectName("centralwidget")

        # 模型选择
        self.btn_selet_model = QtWidgets.QPushButton(self.centralwidget)
        self.btn_selet_model.setGeometry(QtCore.QRect(150, 810, 70, 70))
        self.btn_selet_model.setStyleSheet("border-image: url(:/detect_button_background/upload.png);")
        self.btn_selet_model.setText("")
        self.btn_selet_model.setObjectName("btn_selet_model")
        self.btn_selet_model.clicked.connect(self.seletModels)

        # 选择图像进行检测
        self.btn_detect_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_detect_img.setGeometry(QtCore.QRect(400, 810, 70, 70))
        self.btn_detect_img.setStyleSheet("border-image: url(:/detect_button_background/images.png);")
        self.btn_detect_img.setText("")
        self.btn_detect_img.setObjectName("btn_detect_img")
        self.btn_detect_img.clicked.connect(self.openImage)

        # 保存结果图像
        self.btn_save_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save_img.setGeometry(QtCore.QRect(900, 810, 70, 70))
        self.btn_save_img.setStyleSheet("border-image: url(:/detect_button_background/save.png);")
        self.btn_save_img.setText("")
        self.btn_save_img.setObjectName("btn_save_img")
        self.btn_save_img.clicked.connect(self.saveImage)

        # 清除结果图像
        self.btn_clear_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_clear_img.setGeometry(QtCore.QRect(1150, 810, 70, 70))
        self.btn_clear_img.setStyleSheet("border-image: url(:/detect_button_background/delete.png);")
        self.btn_clear_img.setText("")
        self.btn_clear_img.setObjectName("btn_clear_img")
        self.btn_clear_img.clicked.connect(self.clearImage)

        # 退出应用
        self.btn_exit_app = QtWidgets.QPushButton(self.centralwidget)
        self.btn_exit_app.setGeometry(QtCore.QRect(1345, 810, 70, 70))
        self.btn_exit_app.setStyleSheet("border-image: url(:/detect_button_background/exit.png);")
        self.btn_exit_app.setText("")
        self.btn_exit_app.setObjectName("btn_exit_app")
        self.btn_exit_app.clicked.connect(self.exitApp)

        # 在btn_detect_img后添加视频检测按钮
        self.btn_detect_video = QtWidgets.QPushButton(self.centralwidget)
        self.btn_detect_video.setGeometry(QtCore.QRect(650, 810, 70, 70))
        self.btn_detect_video.setStyleSheet("border-image: url(:/detect_button_background/images.png);")
        self.btn_detect_video.setText("")
        self.btn_detect_video.setObjectName("btn_detect_video")
        self.btn_detect_video.clicked.connect(self.detectVideo)

        # 呈现原始图像
        self.label_show_yuanshi = QtWidgets.QLabel(self.centralwidget)
        self.label_show_yuanshi.setGeometry(QtCore.QRect(50, 80, 650, 650))
        self.label_show_yuanshi.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show_yuanshi.setObjectName("label_show_yuanshi")

        # 呈现结果图像
        self.label_show_jieguo = QtWidgets.QLabel(self.centralwidget)
        self.label_show_jieguo.setGeometry(QtCore.QRect(800, 80, 650, 650))
        self.label_show_jieguo.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show_jieguo.setObjectName("label_show_jieguo")

        # 呈现功能按键
        self.label_show_button = QtWidgets.QLabel(self.centralwidget)
        self.label_show_button.setGeometry(QtCore.QRect(0, 780, 1501, 170))
        self.label_show_button.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show_button.setText("")
        self.label_show_button.setObjectName("label_show_button")

        # 修改文本框的宽度和位置
        text_width = 161  # 恢复原来的宽度

        # 编写模型加载
        self.edit_selet_model = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_selet_model.setGeometry(QtCore.QRect(100, 890, text_width, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(20)  # 调整字体大小使文字更协调
        self.edit_selet_model.setFont(font)
        self.edit_selet_model.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.edit_selet_model.setObjectName("edit_selet_model")

        # 编写图像加载
        self.edit_detect_img = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_detect_img.setGeometry(QtCore.QRect(355, 890, text_width, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(20)
        self.edit_detect_img.setFont(font)
        self.edit_detect_img.setObjectName("edit_detect_img")

        # 编写图像保存
        self.edit_save_img = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_save_img.setGeometry(QtCore.QRect(855, 890, text_width, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(20)
        self.edit_save_img.setFont(font)
        self.edit_save_img.setObjectName("edit_save_img")

        # 编写图像清除
        self.edit_clear_img = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_clear_img.setGeometry(QtCore.QRect(1105, 890, text_width, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(20)
        self.edit_clear_img.setFont(font)
        self.edit_clear_img.setObjectName("edit_clear_img")

        # 编写应用退出
        self.edit_exit_app = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_exit_app.setGeometry(QtCore.QRect(1300, 890, text_width, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(20)
        self.edit_exit_app.setFont(font)
        self.edit_exit_app.setObjectName("edit_exit_app")

        # 添加视频检测
        self.edit_detect_video = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_detect_video.setGeometry(QtCore.QRect(605, 890, text_width, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(20)
        self.edit_detect_video.setFont(font)
        self.edit_detect_video.setObjectName("edit_detect_video")

        # 标题
        self.label_show_title = QtWidgets.QLabel(self.centralwidget)
        self.label_show_title.setGeometry(QtCore.QRect(200, 10, 1100, 60))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(28)
        self.label_show_title.setFont(font)
        self.label_show_title.setStyleSheet("")
        self.label_show_title.setObjectName("label_show_title")

        self.label_show_button.raise_()
        self.btn_selet_model.raise_()
        self.btn_detect_img.raise_()
        self.btn_detect_video.raise_()
        self.btn_save_img.raise_()
        self.btn_clear_img.raise_()
        self.btn_exit_app.raise_()
        self.label_show_title.raise_()
        self.label_show_yuanshi.raise_()
        self.label_show_jieguo.raise_()
        self.edit_selet_model.raise_()
        self.edit_detect_img.raise_()
        self.edit_save_img.raise_()
        self.edit_clear_img.raise_()
        self.edit_exit_app.raise_()
        self.edit_detect_video.raise_()

        # 主窗口
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self.centralwidget)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_show_title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:28pt; font-weight:600; color:#ffffff;\">基于YOLOv8的检测演示软件</span></p></body></html>"))
        self.label_show_yuanshi.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">原始图像</span></p></body></html>"))
        self.label_show_jieguo.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">检测图像</span></p></body></html>"))
        self.edit_selet_model.setText(_translate("MainWindow", "  模型加载"))
        self.edit_detect_img.setText(_translate("MainWindow", "  图像加载"))
        self.edit_save_img.setText(_translate("MainWindow", "  图像保存"))
        self.edit_clear_img.setText(_translate("MainWindow", "  图像清除"))
        self.edit_exit_app.setText(_translate("MainWindow", "  应用退出"))
        self.edit_detect_video.setText(_translate("MainWindow", "  视频检测"))

    # 模型选择函数
    def seletModels(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.btn_selet_model, '选择weights文件', '.', '权重文件(*.pt)')
        if not self.openfile_name_model:
            QMessageBox.warning(self, "Warning:", "打开权重失败", buttons=QMessageBox.Ok, )
        else:
            print('加载weights文件地址为：' + str(self.openfile_name_model))
            QMessageBox.information(self, u"Notice", u"权重打开成功", buttons=QtWidgets.QMessageBox.Ok)

    # 图像选择函数
    def openImage(self):
        try:
            # 选择图片文件
            fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', '图像文件(*.jpg)')
            if not fname:  # 如果用户没有选择文件就返回
                return

            self.fname = fname

            # 立即显示原始图片
            pixmap = QtGui.QPixmap(fname)
            self.label_show_yuanshi.setPixmap(pixmap)
            self.label_show_yuanshi.setScaledContents(True)

            # 强制更新界面显示
            QApplication.processEvents()

            # 检查是否已加载模型，如果已加载则行检测
            if hasattr(self, 'openfile_name_model'):
                self.detect_image()
            else:
                QMessageBox.warning(self, "Warning:", "请先加载模型文件", buttons=QMessageBox.Ok)

        except Exception as e:
            QMessageBox.warning(self, "Error:", f"打开图片时发生错误：{str(e)}", buttons=QMessageBox.Ok)

    def detect_image(self):
        try:
            # 进行模型检测
            model = YOLO(self.openfile_name_model)
            results = model.predict(source=self.fname)
            annotated_frame = results[0].plot()

            # 将检测结果转换为QImage并显示
            height, width, channel = annotated_frame.shape
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(annotated_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.qImg = qimage

            # 显示检测结果
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.label_show_jieguo.setPixmap(pixmap)
            self.label_show_jieguo.setScaledContents(True)

        except Exception as e:
            QMessageBox.warning(self, "Error:", f"模型检测时发生错误：{str(e)}", buttons=QMessageBox.Ok)

    # 图像保存函数
    def saveImage(self):
        fd, _ = QFileDialog.getSaveFileName(self, "保存图片", ".", "*.jpg")
        self.qImg.save(fd)

    # 图像清除函数
    def clearImage(self, stopp):
        result = QMessageBox.question(self, "Warning:", "是否清除本次检测结果", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            self.label_show_yuanshi.clear()
            self.label_show_jieguo.clear()
        else:
            stopp.ignore()

    # 应用退出函数
    def exitApp(self, event):
        result = QMessageBox.question(self, "Notice:", "您真的要退出此应用吗", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            QApplication.instance().quit()

    # 添加视频检测函数
    def detectVideo(self):
        try:
            # 选择视频文件
            video_path, _ = QFileDialog.getOpenFileName(self, "选择视频", ".", "视频文件(*.mp4 *.avi)")
            if not video_path:
                return

            # 检查是否已加载模型
            if not hasattr(self, 'openfile_name_model'):
                QMessageBox.warning(self, "Warning:", "请先加载模型文件", buttons=QMessageBox.Ok)
                return

            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error:", "无法打开视频文件", buttons=QMessageBox.Ok)
                return

            # 加载模型
            model = YOLO(self.openfile_name_model)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 调整帧大小以适应显示区域
                frame = cv2.resize(frame, (700, 700))

                # 显示原始帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_frame = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.label_show_yuanshi.setPixmap(QtGui.QPixmap.fromImage(qt_frame))

                # 进行检测
                results = model.predict(source=frame)
                annotated_frame = results[0].plot()

                # 显示检测结果
                h, w, ch = annotated_frame.shape
                bytes_per_line = ch * w
                qt_detected = QtGui.QImage(annotated_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.label_show_jieguo.setPixmap(QtGui.QPixmap.fromImage(qt_detected))

                # 处理界面事件，使界面保持响应
                QApplication.processEvents()

            cap.release()

        except Exception as e:
            QMessageBox.warning(self, "Error:", f"视频检测时发生错误：{str(e)}", buttons=QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
