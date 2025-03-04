import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tp
from torchvision.datasets import ImageNet


class YoloV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.ZeroPad2d((2, 3, 2, 3)),  # input = 449 x 449 x 3, padding = 2 (same)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2),                # 224 x 224 x 64 (output)
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),                                             # 112 x 112 x 64

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), padding_mode='same'),   # 112 x 112 x 192
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),                                             # 56 x 56 x 192

            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1)),                       # 56 x 56 x 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding_mode='same'),  # 56 x 56 x 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),                       # 56 x 56 x 256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 56 x 56 x 512
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),                                             # 28 x 28 x 512

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),                       # 28 x 28 x 256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 28 x 28 x 512
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),                       # 28 x 28 x 256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 28 x 28 x 512
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),                       # 28 x 28 x 256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 28 x 28 x 512
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),                       # 28 x 28 x 256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding_mode='same'),  # 28 x 28 x 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)),                       # 28 x 28 x 512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding_mode='same'), # 28 x 28 x 1024
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),                                             # 14 x 14 x 1024

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),                      # 14 x 14 x 512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding_mode='same'), # 14 x 14 x 1024
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)),                      # 14 x 14 x 512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding_mode='same'), # 14 x 14 x 1024
        )
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding_mode='same'),            # 14 x 14 x 1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding_mode='same', stride=2),  # 7 x 7 x 1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding_mode='same'),            # 7 x 7 x 1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding_mode='same'),            # 7 x 7 x 1024
            # TODO add connected layer from [7 x 7 x 1024] to [4096]
            # TODO add connected layer from [4096] to [7 x 7 x 30]
            # bilinear transformation?
        )

    def forward(self, x):
        pass


def visualize(dataloader):
    samples, ids = next(iter(dataloader))
    print(f"Feature shape: {samples[0].size()}")
    print(f"Label shape: {ids[0].size()}")

    for i in range(dataloader.batch_size):
        img = samples[i].squeeze()
        img = tp.ToPILImage()(img)
        id = ids[i].item()
        label = dataloader.dataset.classes[id][0]
        plt.imshow(img)
        plt.suptitle(f"[{id}] {label}")
        plt.show()


def train():
    transforms = tp.Compose([
        tp.ToTensor(),
        tp.Resize(size=(224, 224)),
    ])
    train_set = ImageNet(root="./dataset", split="train", transform=transforms)
    val_set = ImageNet(root="./dataset", split="val", transform=transforms)
    print("Classes:", train_set.classes)
    train_data = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    val_data = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=4)
    visualize(train_data)

train()
