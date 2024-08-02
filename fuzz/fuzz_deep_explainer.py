#!/usr/bin/python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
    python3 -m coverage run fuzz_deep_explainer.py -atheris_runs=10
    coverage report -m --omit=../fuzz/config*
"""

import atheris
import numpy as np
import sys
import itertools

default_path = "../plugins"
sys.path.append(default_path)

# This tells Atheris to instrument all functions in the library
with atheris.instrument_imports(include=["intel_ai_safety.explainer.attributions.attributions"]):
    from intel_ai_safety.explainer.attributions.attributions import deep_explainer

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
torch.manual_seed(0)

batch_size = 128
num_epochs = 1
device = torch.device('cpu')


# MockNet class to replace the actual Net class for faster testing
class MockNet(nn.Module):
    def __init__(self):

        super(MockNet, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(784, 10),  # Assuming input is a flattened MNIST image (28x28)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = self.fc_layers(x)
        return x


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
        transforms.ToTensor()])), batch_size=batch_size, shuffle=True)


@atheris.instrument_func
def test_deep_explainer(input_bytes):

    fdp = atheris.FuzzedDataProvider(input_bytes)
    # Generate random data based on the fuzzed input
    num_background = fdp.ConsumeIntInRange(1, 5)
    num_targets = fdp.ConsumeIntInRange(1, 5)
    num_classes = fdp.ConsumeIntInRange(2, 10)
    # The model expects images of shape (batch_size, channels, height, width)
    # For MNIST, this is typically (batch_size, 1, 28, 28)
    # Generate random images with the same shape
    background_images = np.random.rand(num_background, 1, 28, 28).astype(np.float32)
    target_images = np.random.rand(num_targets, 1, 28, 28).astype(np.float32)
    # The labels should be a list of strings, one for each class
    labels = [f"Class {i}" for i in range(num_classes)]
    # Use the mocked model instead of the actual Net to speedup the test
    model = MockNet().to(device)
    # Evaluate the model with a smaller subset of the test data to speedup the test
    model.eval()
    test_loss = 0
    correct = 0
    y_true = torch.empty(0)
    y_pred = torch.empty((0, 10))
    X_test = torch.empty((0, 1, 28, 28))

    with torch.no_grad():
        for data, target in itertools.islice(test_loader, 10):  # Limit the number of batches
            data, target = data.to(device), target.to(device)
            output = model(data)
            X_test = torch.cat((X_test, data))
            y_true, y_pred = torch.cat((y_true, target)), torch.cat((y_pred, output))

            test_loss += F.nll_loss(output.log(), target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    classes = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    # Use the generated background and target images for the deep explainer
    deep_explainer(model, torch.tensor(background_images), torch.tensor(target_images), classes)

    return


atheris.Setup(sys.argv, test_deep_explainer)
atheris.Fuzz()
