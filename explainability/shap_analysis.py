import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

class VeinSHAPAnalyzer:
    def __init__(self, model, background_data, device='cuda'):
        self.model = model
        self.device = device
        self.background = background_data.to(device)
        self.model.eval()

    def create_explainer(self, method='gradient'):
        if method == 'gradient':
            self.explainer = shap.GradientExplainer(self.model, self.background)
        else:
            self.explainer = shap.DeepExplainer(self.model, self.background)
        return self.explainer

    def explain_instance(self, input_image):
        input_image = input_image.to(self.device)
        shap_values = self.explainer.shap_values(input_image)
        with torch.no_grad():
            pred = self.model(input_image).argmax(dim=1).item()
        if isinstance(shap_values, list):
            shap_values = shap_values[pred]
        return shap_values, pred

    def visualize_shap_instance(self, input_image, shap_values, save_path=None):
        shap_map = np.abs(shap_values).mean(axis=1)[0] if shap_values.ndim == 4 else np.abs(shap_values[0])
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(input_image[0, 0].cpu(), cmap='gray'); plt.title("Original")
        plt.subplot(1, 2, 2); plt.imshow(shap_map, cmap='hot'); plt.title("SHAP Importance")
        if save_path: plt.savefig(save_path)
        plt.show()