import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import hog
from skimage import exposure
from tqdm import tqdm

from util import slidingWindow


class Detector:
    def __init__(self, image_size=(2048, 2048), window_size=(64, 64)):
        """
        初始化Detector类
        :param image_size: 图像大小，默认2048x2048
        :param window_size: 提取HOG特征的窗口大小，默认为64x64。
        """
        self.image_size = image_size
        self.window_size = window_size

    @staticmethod
    def read_labels(label_file):
        """
        读取标签文件，返回交通标志的位置
        :param label_file: 标签文件路径
        :return: 标注框列表 [(class_id, x_center, y_center, width, height), ...]
        """
        with open(label_file, 'r') as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # 我们忽略类别ID
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append((class_id, x_center, y_center, width, height))
        return labels

    def get_samples(self, image_path, label_file, window_size=(64, 64)):
        """
        从图像和标签文件中提取正负样本
        :param image_path: 图像路径
        :param label_file: 标签文件路径
        :param window_size: HOG窗口大小，默认64x64
        :return: 正样本（交通标志区域）和负样本（随机区域）
        """
        # 读取图像
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        # 读取标签
        labels = self.read_labels(label_file)

        positive_samples = []
        negative_samples = []

        # 提取正样本：交通标志区域
        for label in labels:
            _, x_center, y_center, width, height = label

            # 将中心点和宽高转换为像素坐标
            x_center_pixel = int(x_center * w)
            y_center_pixel = int(y_center * h)
            width_pixel = int(width * w)
            height_pixel = int(height * h)

            # 获取交通标志区域的左上角和右下角
            x1 = int(x_center_pixel - width_pixel / 2)
            y1 = int(y_center_pixel - height_pixel / 2)
            x2 = int(x_center_pixel + width_pixel / 2)
            y2 = int(y_center_pixel + height_pixel / 2)

            # 确保边界框不会超出图像边界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 提取正样本（交通标志区域的子图）
            crop_img = image[y1:y2, x1:x2]
            positive_samples.append(crop_img)

        # 提取负样本：从图像中随机选取不包含交通标志的区域
        num_negative_samples = len(positive_samples)

        for _ in range(num_negative_samples):
            # 随机生成一个窗口位置
            x1 = np.random.randint(0, w - window_size[0])
            y1 = np.random.randint(0, h - window_size[1])
            x2 = x1 + window_size[0]
            y2 = y1 + window_size[1]

            # 提取负样本区域
            crop_img = image[y1:y2, x1:x2]

            # 只在该区域尺寸足够时才提取HOG特征
            hog_features = self.extract_hog(crop_img)

            if hog_features is not None:
                negative_samples.append(crop_img)

        return positive_samples, negative_samples

    def extract_hog(self, image, window_size=(64, 64)):
        """
        提取HOG特征，并确保图像已经被调整为指定的大小。
        :param image: 输入图像
        :param window_size: 确保图像被缩放到这个尺寸
        :return: HOG特征向量
        """
        # 确保图像被调整为 window_size 大小
        image_resized = cv2.resize(image, (window_size[1], window_size[0]))  # (宽, 高)

        # 转换为灰度图
        gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        # 提取HOG特征
        fd, hog_image = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

        return fd

    def prepare_data(self, images_folder, labels_folder, window_size=(64, 64)):
        """
        准备所有图像和标签的数据集
        :param images_folder: 图像文件夹路径
        :param labels_folder: 标签文件夹路径
        :return: 所有正样本和负样本
        """
        positive_samples = []
        negative_samples = []

        for filename in os.listdir(images_folder):
            if filename.endswith('.jpg'):
                image_path = os.path.join(images_folder, filename)
                label_file = os.path.join(labels_folder, filename.replace('.jpg', '.txt'))

                if not os.path.exists(label_file):
                    continue

                # 获取正负样本
                pos_samples, neg_samples = self.get_samples(image_path, label_file, window_size)
                positive_samples.extend(pos_samples)
                negative_samples.extend(neg_samples)

        return positive_samples, negative_samples

    def train_svm(self, positive_samples, negative_samples):
        """
        训练SVM分类器
        :param positive_samples: 正样本
        :param negative_samples: 负样本
        :return: 训练好的SVM模型
        """
        # 提取正样本和负样本的HOG特征
        positive_hogs = [self.extract_hog(img) for img in tqdm(positive_samples, desc="提取正样本HOG特征")]
        negative_hogs = [self.extract_hog(img) for img in tqdm(negative_samples, desc="提取负样本HOG特征")]

        # 将正负样本合并，正样本标签为1，负样本标签为0
        X = np.array(positive_hogs + negative_hogs)
        y = np.array([1] * len(positive_hogs) + [0] * len(negative_hogs))

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 创建SVM分类器
        svm_classifier = SVC(kernel='rbf', C=1000)  # 可以选择不同的内核

        # 使用 tqdm 显示训练过程
        # tqdm 在 fit 函数内不直接支持，我们可以通过自定义训练过程来显示进度

        svm_classifier.fit(X_train, y_train)

        # 在测试集上评估模型
        y_pred = svm_classifier.predict(X_test)
        print("SVM分类器性能：")
        print(classification_report(y_test, y_pred))

        return svm_classifier

    def predict(self, svm_classifier, image_path, threshold=0.5, init_size=(64, 64), x_overlap=0.5, y_step=0.05, scale=1.5):
        """
        使用训练好的SVM分类器进行预测，采用滑动窗口扩张方法检测目标物体
        :param svm_classifier: 训练好的SVM模型
        :param image_path: 测试图像路径
        :param threshold: 决策函数的阈值，默认为0.5
        :param init_size: 初始窗口大小，默认为(64, 64)
        :param x_overlap: 水平方向的重叠度，默认为0.5
        :param y_step: 垂直方向的步长，默认为0.05
        :param scale: 窗口扩展系数，默认为1.5
        :return: 检测结果，包含检测框的位置
        """
        # 读取图像
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        # 使用滑动窗口扫描图像
        windows = slidingWindow((w, h), init_size=init_size, x_overlap=x_overlap, y_step=y_step, scale=scale)

        # 存储检测到的位置
        detection_results = []

        for (x1, y1, x2, y2) in windows:
            # 提取当前窗口的区域
            window = image[y1:y2, x1:x2]

            # 提取HOG特征
            fd = self.extract_hog(window)

            # 使用SVM进行预测
            decision_value = svm_classifier.decision_function([fd])
            print(decision_value)
            # 如果决策值大于阈值，则认为是正类（交通标志）
            if decision_value > threshold:
                detection_results.append((x1, y1, x2, y2))

        return detection_results


# 示例使用
if __name__ == "__main__":
    images_folder = "./mini_test/images"  # 替换为图像文件夹路径
    labels_folder = "./mini_test/labels"  # 替换为标签文件夹路径

    detector = Detector(image_size=(2048, 2048), window_size=(256, 256))

    # 准备数据
    positive_samples, negative_samples = detector.prepare_data(images_folder, labels_folder)

    # 训练SVM分类器
    svm_classifier = detector.train_svm(positive_samples, negative_samples)

    # 使用训练好的SVM模型进行预测
    image_path = "2.jpg"  # 替换为要测试的图像路径
    detection_results = detector.predict(svm_classifier, image_path)

    # 打印检测结果
    print("检测到的交通标志位置：", detection_results)
