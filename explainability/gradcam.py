import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        weights = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        cam = (weights * self.activations[0]).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.cpu().numpy(), output, target_class

    def visualize(self, input_image, cam, save_path=None):
        cam_resized = cv2.resize(cam, (input_image.shape[-2], input_image.shape[-1]))
        heatmap = cm.jet(cam_resized)[:, :, :3]
        img_np = input_image[0, 0].cpu().numpy()
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_rgb = np.stack([img_norm] * 3, axis=-1)
        overlay = 0.5 * img_rgb + 0.5 * heatmap
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(img_norm, cmap='gray'); plt.title("Original")
        plt.subplot(1, 2, 2); plt.imshow(overlay); plt.title("Grad-CAM Overlay")
        if save_path: plt.savefig(save_path)
        plt.show()