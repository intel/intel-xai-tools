class XGradCAM:
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
        rgb_img = np.float32(image) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.to(self.device)
        target_layer_edit = self.model
        for layer in self.targetLayer.split('.')[1:]:
            target_layer_edit = getattr(target_layer_edit, layer)
        target_layers = [target_layer_edit]
        if self.targetClass is None:
            targets = None
        else:
            targets = [ClassifierOutputTarget(self.targetClass)]
        cam = XGradCAM(self.model, target_layers, use_cuda=torch.cuda.is_available())
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

def xgradcam(model, targetLayer, targetClass, image, dims, device):
    return XGradCAM(model, targetLayer, targetClass, image, dims, device)
