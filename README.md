渣土车（muck_truck）YOLOv11 检测系统  渣土车检测数据集

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b457ed1408b4f96bd6f34ec82077857.png)
渣土车检测数据集，2502张，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/835bc4cfe1254cc6b644a1475f21b23b.png)
yolo和voc两种标注方式
1类，标注数量：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5b0a8ba15c6f4cc2968e6e7f664aa5f3.png)

muck_truck: 4240
image num: 2502![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f4d7db024d944153a370b0e83a8d32b6.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b8f2f8030c774fbab81d14c346e50bdf.png)

2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e2ca7b2999fe42bf9edcdaa79352f761.png)

3.qt界面：运行界面采用qt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bc81a289709f4398abe2a6610cb193b2.png)
# 渣土车（muck_truck）YOLOv11检测系统全套代码
## 1、数据集yaml配置文件 `muck_truck.yaml`
```yaml
path: ./muck_truck_dataset
train: images/train
val: images/val
test: images/test

names:
  0: muck_truck
```

## 2、模型训练代码 train.py
```python
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv11n检测模型
    model = YOLO("yolov11n.pt")
    # 开始训练
    train_result = model.train(
        data="muck_truck.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project="muck_truck_result",
        name="yolov11_mucktruck",
        save=True,
        val=True
    )
    print("训练结束，最佳权重：muck_truck_result/yolov11_mucktruck/weights/best.pt")
```

## 3、图像/视频推理代码 predict.py
```python
from ultralytics import YOLO
import cv2

model = YOLO("muck_truck_result/yolov11_mucktruck/weights/best.pt")

def detect_img(img_path):
    res = model.predict(img_path, conf=0.4)
    for r in res:
        img = r.plot()
    cv2.imwrite("detect_result.jpg", img)
    return img

def detect_video(video_path):
    model.predict(video_path, save=True, conf=0.4)

if __name__ == "__main__":
    detect_img("test.jpg")
```

## 4、PyQt5可视化界面完整代码（对应截图里的系统界面）
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class MuckTruckWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的渣土车检测")
        self.setGeometry(100, 100, 1200, 800)
        self.model = YOLO("muck_truck_result/yolov11_mucktruck/weights/best.pt")

        # 图片显示label
        self.img_label = QLabel(self)
        self.img_label.setGeometry(20, 100, 600, 500)
        self.img_label.setStyleSheet("border:1px solid black;")

        # 按钮
        self.btn_open = QPushButton("导入图片", self)
        self.btn_open.move(650,120)
        self.btn_open.clicked.connect(self.load_image)

        self.btn_save = QPushButton("保存结果", self)
        self.btn_save.move(650,300)
        self.btn_save.clicked.connect(self.save_result)

        # 结果表格
        self.table = QTableWidget(self)
        self.table.setGeometry(20,620,800,150)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标"])
        self.result_img = None

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName()
        if not fname:
            return
        img = cv2.imread(fname)
        results = self.model.predict(fname, conf=0.4)
        self.table.setRowCount(0)
        for idx, r in enumerate(results):
            for box in r.boxes:
                cls = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = xyxy
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row,0,QTableWidgetItem(str(row+1)))
                self.table.setItem(row,1,QTableWidgetItem(fname))
                self.table.setItem(row,2,QTableWidgetItem(cls))
                self.table.setItem(row,3,QTableWidgetItem(f"{conf:.2f}"))
                self.table.setItem(row,4,QTableWidgetItem(f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"))
        img_draw = results[0].plot()
        self.result_img = img_draw
        img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        h,w,c = img_rgb.shape
        qimg = QImage(img_rgb.data,w,h,c*w,QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(),Qt.KeepAspectRatio))

    def save_result(self):
        if self.result_img is not None:
            cv2.imwrite("detect_save.jpg", self.result_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MuckTruckWindow()
    win.show()
    sys.exit(app.exec_())
```

## 环境安装
```bash
pip install ultralytics opencv-python pyqt5
```

## 标签
`muck_truck`，中文：渣土车

## 应用场景
1. 工地出入口、城市道路摄像头实时监测渣土车通行
2. 智慧城管，自动识别渣土车，统计车流量
3. 渣土车违规管控、超载/密闭监测前置识别
4. 毕业设计、目标检测算法研究

