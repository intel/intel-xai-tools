from ..utils.model.model_framework import is_tf_model, is_pt_model, raise_unknown_model_error


class GradCAM:
    """GradCAM base class. Depending on the model framework, GradCAM is a superclass to TFGradCAM or XGradCAM.
    Note that EiganCAM (only supports PyTorch) is not included yet.
    """
    def __new__(cls, model, *args):    
        if is_tf_model(model):
            return super().__new__(TFGradCAM)
        elif is_pt_model(model):
            return super().__new__(XGradCAM)
        else:
            raise_unknown_model_error()
        

class TFGradCAM(GradCAM):
    '''
    Holds the calculations for the gradient-weighted class activation mapping (gradCAM) of a 
    given image and TensorFlow CNN.

    Args:
      model (tf.keras.functional): the CNN used for classification 
      target_layer (tf.keras.KerasLayer): the convolution layer that you want to analyze (usually the last) 
      target_class (int): the index of the target class
      image (numpy.ndarray): image to be analyzed with a shape (h,w,c)

        
    Attributes:
      model: the CNN being used
      target_layer: the target convolution being used 
      target_class: the target class being used
      image: the image being used
      dims: the dimensions of the image being used
      gradcam: the result of the gradCAM calculation from the model's target_layer on the image


    Methods:
      visualize: superimpose the gradCAM result on top of the original image

    Reference:
      https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb 
    '''
    def __init__(self, model, target_layer, target_class, image):
        
        self.model = model
        self.target_layer = target_layer
        self.target_class = target_class
        self.image = image
        self.dims = (image.shape[0], image.shape[1])
        
        self.gradcam = self._get_gradcam()
    
    def _get_gradcam(self):
        import numpy as np
        import tensorflow as tf
        import cv2

        last_conv_layer_model = tf.keras.Model(self.model.inputs, self.target_layer.output)
        classifier_input = tf.keras.Input(shape=self.target_layer.output.shape[1:])
        x = classifier_input
        
        # get the last conv layer and all the proceeding layers  
        last_layers = []
        for layer in reversed(self.model.layers):
            last_layers.append(layer.name)
            if 'conv' in layer.name or 'pool' in layer.name:
                break
        
        # create the classifier model to get the gradient for the
        # target class
        for layer_name in reversed(last_layers):
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        
        with tf.GradientTape() as tape:
            inputs = self.image[np.newaxis, ...]
            last_conv_layer_output = last_conv_layer_model(inputs)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_class_channel = preds[:, self.target_class]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
        # Average over all the filters to get a single 2D array
        gradcam = np.mean(last_conv_layer_output, axis=-1)
        # Clip the values (equivalent to applying ReLU)
        # and then normalise the values
        gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
        return cv2.resize(gradcam, self.dims)
        
        

    def visualize(self):
        import matplotlib.pyplot as plt
        
        plt.imshow(self.image)
        plt.imshow(self.gradcam, alpha=0.5)
        plt.axis('off')


class XGradCAM(GradCAM):
    '''
    Holds the calculations for the axiom-based gradient-weighted class activation mapping (XgradCAM) of a 
    given image and PyTorch CNN.

    Args:
      model (torch.nn.Module): the CNN used for classification 
      target_layer (torch.nn.modules.container.Sequential): the convolution layer that you want to analyze (usually the last) 
      target_class (int): the index of the target class
      image (numpy.ndarray): image to be analyzed with a shape (h,w,c)
      dims (tuple of ints): dimension of image (h, w)
      device (torch.device): torch.device('cpu') or torch.device('gpu') for PyTorch optimizations

        
    Attributes:
      model: the CNN being used
      target_layer: the target convolution being used 
      target_class: the target class being used
      image: the image being used
      dims: the dimensions of the image being used
      device: device being used by PyTorch

    Methods:
      visualize: superimpose the gradCAM result on top of the original image

    Reference:
       https://github.com/jacobgil/pytorch-grad-cam
    '''
    def __init__(self, model, targetLayer, targetClass, image, dims, device):

        # set any frozen layers to trainable
        # gradcam cannot be calculated without it
        for param in model.parameters():
            if not param.requires_grad:
                param.requires_grad = True

        self.model = model
        self.targetLayer = targetLayer
        self.targetClass = targetClass
        self.image = image
        self.dims = dims
        self.device = device

    def visualize(self):
        from pytorch_grad_cam import XGradCAM, GuidedBackpropReLUModel
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
        import torch
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        self.model.eval().to(self.device)

        image = cv2.resize(self.image, self.dims)
        # convert to rgb if image is grayscale
        converted = False
        if len(image.shape) == 2:
            converted = True 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        rgb_img = np.float32(image) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.to(self.device)
        
        self.targetLayer = [self.targetLayer]
        
        if self.targetClass is None:
            targets = None
        else:
            targets = [ClassifierOutputTarget(self.targetClass)]

        cam = XGradCAM(self.model, self.targetLayer, use_cuda=torch.cuda.is_available())

        # convert back to grayscale if that is the initial dim
        if converted:
            input_tensor = input_tensor[:, 0:1, :, :]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=False,
                            eigen_smooth=False)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=self.model, use_cuda=torch.cuda.is_available())
        gb = gb_model(input_tensor, target_category=None)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        fig = plt.figure(figsize=(10, 7))
        rows = 1
        columns = 3

        fig.add_subplot(rows, columns, 1)
        plt.imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("XGradCAM")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(cv2.cvtColor(gb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Guided backpropagation")

        fig.add_subplot(rows, columns, 3)
        plt.imshow(cv2.cvtColor(cam_gb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Guided XGradCAM")

        print("XGradCAM, Guided backpropagation, and Guided XGradCAM are generated. ")



class EigenCAM:

    '''
    Holds the calculations for the eigan-based gradient-weighted class activation mapping (EiganCAM) of a 
    given image and PyTorch CNN for object detection.

    Args:
      model (torch.nn.Module): the CNN used for classification 
      target_layer (torch.nn.modules.container.Sequential): the convolution layer that you want to analyze (usually the last) 
      boxes (list): list of coordinates where the object is detected
      classes (list): list of classes that are predicted from boxes
      colors (list): list of colors corresponding to the classes
      reshape (function): the reshape transformation function responsible for processing the output tensors. Can be None
        if not needed for particular model (such as YOLO)
      image (numpy.ndarray): image to be analyzed with a shape (h,w,c)
      device (torch.device): torch.device('cpu') or torch.device('gpu') for PyTorch optimizations

        
    Attributes:
      model: the CNN being used
      target_layer: the target convolution being used 
      boxes: the list of coordinates being used
      classes: the list of classes being used
      colors: the list of colors being used for the classes
      reshape: the transformation function being used to process model output
      image: the image being used
      device: device being used by PyTorch

    Methods:
      visualize: superimpose the EiganCAM  result on top of the original image

    Reference:
       https://github.com/jacobgil/pytorch-grad-cam 
    '''

    def __init__(self, model, targetLayer, boxes, classes, colors, reshape, image, device):
        self.model = model
        self.targetLayer = targetLayer
        self.boxes = boxes
        self.classes = classes
        self.colors = colors
        self.reshape = reshape
        self.image = image
        self.device = device

    def visualize(self):
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, scale_cam_image
        import torchvision
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        from IPython.display import display

        self.model.eval().to(self.device)

        rgb_img = np.float32(self.image) / 255
        transform = torchvision.transforms.ToTensor()
        input_tensor = transform(rgb_img)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        self.targetLayer = [self.targetLayer]

        if self.reshape is None:
            cam = EigenCAM(self.model, self.targetLayer, use_cuda=torch.cuda.is_available())
        else:
            cam = EigenCAM(self.model, self.targetLayer, use_cuda=torch.cuda.is_available(),
                           reshape_transform=self.reshape)
        targets = []
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=False,
                            eigen_smooth=False)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in self.boxes:
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(rgb_img, renormalized_cam, use_rgb=True)
        for i, box in enumerate(self.boxes):
            color = self.colors[i]
            cv2.rectangle(
                eigencam_image_renormalized,
                (box[0], box[1]),
                (box[2], box[3]),
                color, 2
            )
            cv2.putText(eigencam_image_renormalized, self.classes[i], (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)

        display(Image.fromarray(np.hstack((cam_image, eigencam_image_renormalized))))

        print("EigenCAM is generated. ")

def tf_gradcam(model, target_layer, target_class, image):
    """
    Generates TFGradCAM object that calculates the gradient-weighted class activation
    mapping of a given image and CNN.

    Args:
      model (tf.keras.Functional): the CNN used for classification 
      target_layer (tf.keras.KerasLayer): the convolution layer that you want to analyze (usually the last) 
      target_class (int): the index of the target class
      image (numpy.ndarray): image to be analyzed with a shape (h,w,c)

    Returns:
      TFGradCAM

    Reference:
       https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb
    """
    return TFGradCAM(model, target_layer, target_class, image)

def xgradcam(model, targetLayer, targetClass, image, dims, device):
    return XGradCAM(model, targetLayer, targetClass, image, dims, device)

def eigencam(model, targetLayer, boxes, classes, colors, reshape, image, device):
    return EigenCAM(model, targetLayer, boxes, classes, colors, reshape, image, device)
