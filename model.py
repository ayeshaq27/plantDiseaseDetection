import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.
    Used as the baseline model before switching to transfer learning.
    """

    def __init__(self, num_classes=38):
        super(BaselineCNN, self).__init__()

        # Feature extractor (convolution layers)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # output: 16x224x224
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # output: 16x112x112

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # output: 32x112x112
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # output: 32x56x56

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # output: 64x56x56
            nn.ReLU(),
            nn.MaxPool2d(2)                                       # output: 64x28x28
        )

        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                         # output: 64*28*28
            nn.Linear(64 * 28 * 28, 128),                         # compress to 128 features
            nn.ReLU(),
            nn.Linear(128, num_classes)                           # final prediction
        )

    def forward(self, x):
        x = self.features(x)                                     # conv layers
        x = self.classifier(x)                                   # FC layers
        return x
