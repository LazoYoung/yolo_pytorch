import os

import cv2
import numpy as np
from cv2.dnn import NMSBoxes
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as tx
from torchvision.datasets import ImageNet

from model import Darknet, Version, Dataset


def detect_with_darknet(darknet, img):
    boxes, scores, id = [], [], []
    height, width = img.shape[0], img.shape[1]
    outputs = darknet.forward(img)
    for output in outputs:
        for feature in output:
            pred_scores = feature[5:]
            pred_class_id = np.argmax(pred_scores)
            conf = feature[4]
            if conf > 0.5:
                center_x, center_y = int(feature[0] * width), int(feature[1] * height),
                w, h = int(feature[2] * width), int(feature[3] * height),
                x, y = int(center_x - w / 2), int(center_y - h / 2),
                boxes.append((x, y, x + w, y + h))
                scores.append(conf)
                id.append(pred_class_id)
    ind = NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)
    pred = [(*boxes[i], scores[i], id[i]) for i in range(len(boxes)) if i in ind]
    return pred


def plot_detection(pred, class_names, img):
    colors = np.random.uniform(low=0, high=255, size=(len(class_names), 3))
    for i, (x1, y1, x2, y2, conf, id) in enumerate(pred):
        text = f"{class_names[id]} / conf={conf:.3f}"
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=colors[id], thickness=2)
        cv2.putText(img, text, org=(x1, y1+30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=colors[id], thickness=2)
    cv2.imshow("YOLO detection", img)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()


def plot_dataset(dataloader):
    samples, ids = next(iter(dataloader))
    print(f"Feature shape: {samples[0].size()}")
    print(f"Label shape: {ids[0].size()}")

    for i in range(dataloader.batch_size):
        img = samples[i].squeeze()
        img = tx.ToPILImage()(img)
        id = ids[i].item()
        label = dataloader.dataset.classes[id][0]
        plt.imshow(img)
        plt.suptitle(f"[{id}] {label}")
        plt.show()


def train():
    transforms = tx.Compose([
        tx.ToTensor(),
        tx.Resize(size=(224, 224)),
    ])
    train_set = ImageNet(root="dataset/imgnet", split="train", transform=transforms)
    val_set = ImageNet(root="dataset/imgnet", split="val", transform=transforms)
    print("Classes:", train_set.classes)
    train_data = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    val_data = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=4)
    plot_dataset(train_data)


def test():
    darknet = Darknet(model=Version.YOLO_V3, data=Dataset.COCO)
    images = get_test_images()
    darknet.inspect_output_layers(images[0])
    for img in images:
        pred = detect_with_darknet(darknet, img)
        plot_detection(pred, darknet.class_names, img)


def get_test_images():
    root = "dataset/test"
    img_ext = ['jpg', 'jpeg', 'png', 'tiff']
    img = []
    for f_name in os.listdir(root):
        if f_name.split('.')[1].lower() in img_ext:
            path = os.path.join(root, f_name)
            img.append(cv2.imread(path))
    return img


test()
