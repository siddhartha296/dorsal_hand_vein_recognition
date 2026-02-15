import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import matplotlib.pyplot as plt

class VeinLIMEAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()

    def batch_predict(self, images):
        # LIME provides (N, H, W, C); model expects (N, C, H, W)
        images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            outputs = self.model(images_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def explain_instance(self, input_image, num_samples=1000):
        # Prepare image for LIME (H, W, C)
        img_np = input_image[0].permute(1, 2, 0).cpu().numpy()
        if img_np.shape[-1] == 1:
            img_np = np.concatenate([img_np]*3, axis=-1) # LIME requires 3 channels
            
        explanation = self.explainer.explain_instance(
            img_np, self.batch_predict, top_labels=5, num_samples=num_samples,
            segmentation_fn=lambda x: slic(x, n_segments=100, compactness=10)
        )
        pred = self.batch_predict(img_np[np.newaxis, ...]).argmax()
        return explanation, pred, None

    def visualize_explanation(self, input_image, explanation, pred_class, save_path=None):
        temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=10, hide_rest=False)
        img_bound = mark_boundaries(temp / temp.max(), mask)
        plt.imshow(img_bound)
        plt.title(f"LIME Explanation for Class {pred_class}")
        if save_path: plt.savefig(save_path)
        plt.show()