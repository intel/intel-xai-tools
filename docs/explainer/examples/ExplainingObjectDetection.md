---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Explaining Object Detection Models

### Objective
The goal of this notebook is to explore various CAM methods for object detection models.
Unlike object classification model, we are restricted to use gradient free CAM method since the outputs from object detection model are typically not differentiable.
Among the gradient free methods, for now, we support EigenCAM which is an extremely fast method.
Let's have a look at two cases: FasterRCNN and YOLO.

### Install Notebook Dependency
```{code-cell} ipython3
!pip install scikit-image
```

### Loading Intel XAI tools CAM Module
```{code-cell} ipython3
from explainer import cam
```
### Loading Notebook Modules
```{code-cell} ipython3
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
```

## Using EigenCAM for FasterRCNN

### Loading the input image
Load the input image as a numpy array in RGB order. 
```{code-cell} ipython3
image = io.imread("https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png")

plt.imshow(image)
plt.show()
```

### Loading the model
Load the trained model depending on how the model was saved. 
```{code-cell} ipython3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = fasterrcnn_resnet50_fpn(pretrained=True).eval() # Let's use FasterRCNN ResNet50 FPN as our model
```
As in the image classification example, choose the target layer (normally the last convolutional layer) to compute CAM for.
```{code-cell} ipython3
targetLayer = model.backbone
```
In the case of FasterRCNN, there is no class name in the output from the model. 
Thus, to print the name of class with corresponding bounding box in the image, we need to specify the name of classes that are trained as a list.
```{code-cell} ipython3
class_labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```
```{code-cell} ipython3
color = np.random.uniform(0, 255, size=(len(class_labels), 3)) # Create a different color for each class
```
We need to process the outputs from the object detection model. 
In the case of FasterRCNN, first, convert the image to a tensor and get the output of the model.
```{code-cell} ipython3
rgb_img = np.float32(image) / 255
transform = torchvision.transforms.ToTensor()
input_tensor = transform(rgb_img)
input_tensor = input_tensor.unsqueeze(0)
output = model(input_tensor)[0]
```
The function below processes the outputs from the model.
We can get the bounding box coordinates, class names, class indices, and box colors of the detected objects with a higher detection score than a threshold value. 
If you use other object detection models than FasterRCNN, you need to make your own function to match the structure of the outputs from this function.
```{code-cell} ipython3
def process_output_fasterrcnn(output, class_labels, color, detection_threshold):
    boxes, classes, labels, colors = [], [], [], []
    box = output['boxes'].tolist()
    name = [class_labels[i] for i in output['labels'].detach().numpy()]
    label = output['labels'].detach().numpy()

    for i in range(len(name)):
        score = output['scores'].detach().numpy()[i]
        if score < detection_threshold:
            continue
        boxes.append([int(b) for b in box[i]])
        classes.append(name[i])
        colors.append(color[label[i]])

    return boxes, classes, colors
```
```{code-cell} ipython3
detection_threshold = 0.9
boxes, classes, colors = process_output_fasterrcnn(output, class_labels, color, detection_threshold)
```
Here is the other important part which differs from the object classification models. 
We need to write a custom reshape transform to get the activations from the model and process them into 2D format. 
In the case of FasterRCNN, the backbone outputs 5 different tenors with different spatial size as an Ordered Dict. 
Thus, we need a custom function which aggregates these image tensors, resizes them to a common shape, and concatenates them. 
If you use other models than FasterRCNN, you might need to write your own custom reshape transform function.
```{code-cell} ipython3
def fasterrcnn_reshape_transform(x):
    target_size = x['pool'].size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations
```

### Visualization
The left image is the computed EigenCAM on the image. 
On the right side, we can renormalize the EigenCAM inside every bounding box and zero outside the bounding boxes.
```{code-cell} ipython3
eigencam = cam.eigencam(model, targetLayer, boxes, classes, colors, fasterrcnn_reshape_transform, image, device)
eigencam.visualize()
```

## Using EigenCAM for YOLO
Computing EigenCAM for YOLO is much simpler than FasterRCNN.

### Loading the input image
Load the input image as a numpy array in RGB order. 
```{code-cell} ipython3
image = io.imread("https://upload.wikimedia.org/wikipedia/commons/f/f1/Puppies_%284984818141%29.jpg")

plt.imshow(image)
plt.show()
```

### Loading the model
Load the trained model depending on how the model was saved. 
```{code-cell} ipython3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # Let's use YOLO5 as our model
```
```{code-cell} ipython3
targetLayer = model.model.model.model[-2]
```
```{code-cell} ipython3
color = np.random.uniform(0, 255, size=(80, 3)) # Create a color list with the length of the number of classes
```
```{code-cell} ipython3
image = cv2.resize(image, (640, 640))
output = model(image)
output = output.pandas().xyxy[0].to_dict()
```
The function below processes the outputs from YOLO model. 
We already have class names in the output from YOLO, so there is no need to specify them.
```{code-cell} ipython3
def process_output_yolo(output, color, detection_threshold):
    boxes, classes, labels, colors = [], [], [], []

    for i in range(len(output["xmin"])):
        confidence = output["confidence"][i]
        if confidence < detection_threshold:
            continue
        xmin = int(output["xmin"][i])
        ymin = int(output["ymin"][i])
        xmax = int(output["xmax"][i])
        ymax = int(output["ymax"][i])
        boxes.append([xmin, ymin, xmax, ymax])
        classes.append(output["name"][i])
        colors.append(color[int(output["class"][i])])

    return boxes, classes, colors
```
```{code-cell} ipython3
detection_threshold = 0.4
boxes, classes, colors = process_output_yolo(output, color, detection_threshold)
```
In the case of YOLO, we don't need a reshape transform function since we’re getting a 2D spatial tensor.
```{code-cell} ipython3
reshape = None
```

### Visualization
```{code-cell} ipython3
eigencam = cam.eigencam(model, targetLayer, boxes, classes, colors, reshape, image, device)
eigencam.visualize()
```

## References
pytorch-grad-cam GitHub tutorial for object detection with FasterRCNN - https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.html

pytorch-grad-cam GitHub tutorial for object detection with YOLO - https://jacobgil.github.io/pytorch-gradcam-book/EigenCAM%20for%20YOLO5.html