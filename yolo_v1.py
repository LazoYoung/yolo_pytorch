import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.dnn import Net, NMSBoxes
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tx
from torchvision.datasets import ImageNet

from model import LocallyConnected2d, Darknet


class YoloV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # input = 3 x 448 x 448
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),     # 64 x 224 x 224 (output)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),                                             # 64 x 112 x 112

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), padding_mode='same'),   # 192 x 112 x 112
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),                                             # 192 x 56 x 56

            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1)),                       # 128 x 56 x 56
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding_mode='same'),  # 256 x 56 x 56
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),                       # 256 x 56 x 56
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 512 x 56 x 56
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),                                             # 512 x 28 x 28

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),                       # 256 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 512 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),                       # 256 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 512 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),                       # 256 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 512 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),                       # 256 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 512 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)),                       # 512 x 28 x 28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding_mode='same'), # 1024 x 28 x 28
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),                                             # 1024 x 14 x 14

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),                      # 512  x 14 x 14
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding_mode='same'), # 1024 x 14 x 14
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),                      # 512  x 14 x 14
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding_mode='same'), # 1024 x 14 x 14
            nn.LeakyReLU(),
        )
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding_mode='same'),            # 1024 x 14 x 14
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding_mode='same', stride=2),  # 1024 x 7 x 7
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding_mode='same'),            # 1024 x 7 x 7
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding_mode='same'),            # 1024 x 7 x 7
            nn.LeakyReLU(),

            LocallyConnected2d(in_channels=1024, out_channels=256, output_size=(7, 7), kernel_size=(3, 3)),     # 256  x 7 x 7
            nn.Dropout2d(p=0.5),

            nn.Flatten(),                                       # 12544
            nn.Linear(in_features=256*7*7, out_features=1715),  # 1715
        )
        # Not sure that it's the correct way
        self.detection_for_train = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Linear(in_features=1715, out_features=1470),
            nn.Unflatten(dim=1, unflattened_size=(7, 7, 30)),
        )
        # Not sure that it's the correct way
        self.detection = nn.Sequential(
            nn.Linear(in_features=1715, out_features=1470),
            nn.Unflatten(dim=1, unflattened_size=(7, 7, 30)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        if self.training:
            # todo compute loss
            pass
        return x


def detect_with_darknet(darknet, img):
    boxes, scores, id = [], [], []
    height, width = img.shape[0], img.shape[1]
    outputs = darknet.forward(img)
    for output in outputs:
        for feature in output:  # feature.shape = 85
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
    train_set = ImageNet(root="./dataset", split="train", transform=transforms)
    val_set = ImageNet(root="./dataset", split="val", transform=transforms)
    print("Classes:", train_set.classes)
    train_data = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    val_data = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=4)
    plot_dataset(train_data)

# train()
test_img = cv2.imread("dataset/test.jpg")
darknet = Darknet()
darknet.inspect_output_layers(test_img)
pred = detect_with_darknet(darknet, test_img)
plot_detection(pred, darknet.class_names, test_img)
