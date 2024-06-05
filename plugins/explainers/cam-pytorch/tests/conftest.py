import pytest
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

torch.manual_seed(0)


@pytest.fixture(scope="session")
def custom_pyt_CNN():
    """
    Creates and trains a simple PyTorch CNN on the mnist dataset.
    Returns the model, the test dataset loader and the class names.

    """
    batch_size = 128
    num_epochs = 1
    device = torch.device("cpu")

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.Dropout(),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(320, 50), nn.ReLU(), nn.Dropout(), nn.Linear(50, 10), nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)
            return x

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output.log(), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("mnist_data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("mnist_data", train=False, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

    class_names = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    y_true = next(iter(test_loader))[1].to(device)
    X_test = next(iter(test_loader))[0].to(device)

    return model, X_test, class_names, y_true


@pytest.fixture(scope="session")
def dog_cat_image():
    """Loads the cat-dog image exampe from imagenet."""
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get("https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png")
    return np.array(Image.open(BytesIO(response.content)))


@pytest.fixture(scope="session")
def imagenet_class_names():
    # load the ImageNet class names as a vectorized mapping function from ids to names
    import shap
    import json

    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with open(shap.datasets.cache(url)) as file:
        class_names = [v[1] for v in json.load(file).values()]
    return class_names


@pytest.fixture(scope="session")
def fasterRCNN():
    """
    Loads the fasterRCNN, the class labels and their corresponding colors
    """

    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    model = fasterrcnn_resnet50_fpn(pretrained=True).eval()
    class_labels = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    color = np.random.uniform(0, 255, size=(len(class_labels), 3))  # Create a different color for each class

    def fasterrcnn_reshape_transform(x):
        target_size = x["pool"].size()[-2:]
        activations = []
        for key, value in x.items():
            activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode="bilinear"))
        activations = torch.cat(activations, axis=1)
        return activations

    return model, class_labels, color, fasterrcnn_reshape_transform
