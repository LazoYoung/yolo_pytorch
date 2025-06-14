from torch import nn

from model import LocallyConnected2d


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
