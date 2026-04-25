"""
GradCAM 可视化模块 - 模型可解释性
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
import cv2
import os


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = target_layer or self._get_last_conv_layer()
        self.hooks = []

    def _get_last_conv_layer(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = (name, module)
        return last_conv[1]

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def register_hooks(self):
        self.hooks.append(self.target_layer.register_forward_hook(self.forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(self.backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_cam(self, input_tensor, target_class=None):
        self.register_hooks()

        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        self.remove_hooks()
        return heatmap.numpy(), target_class

    def generate_overlay(self, image, heatmap, alpha=0.4):
        if isinstance(image, Image.Image):
            image = np.array(image)

        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        jet = cm.get_cmap('jet')
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = (jet_heatmap * 255).astype(np.uint8)
        jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        output = cv2.addWeighted(image, 1 - alpha, jet_heatmap, alpha, 0)
        return output


def visualize_prediction(image_path, model, transform, class_names, output_dir=None):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    gradcam = GradCAM(model)
    heatmap, pred_class = gradcam.generate_cam(img_t)
    overlay = gradcam.generate_overlay(img, heatmap)

    pred_label = class_names[pred_class]
    confidence = torch.softmax(model(img_t), dim=1)[0, pred_class].item()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        result_path = os.path.join(output_dir, f'gradcam_{os.path.basename(image_path)}')
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f'原始图片')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title(f'GradCAM 热力图')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'叠加结果\n{pred_label} ({confidence:.2%})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'result_path': result_path,
            'prediction': pred_label,
            'confidence': confidence,
            'class_index': pred_class
        }
    else:
        return {
            'heatmap': heatmap,
            'overlay': overlay,
            'prediction': pred_label,
            'confidence': confidence,
            'class_index': pred_class
        }


def batch_visualize(image_paths, model, transform, class_names, output_dir):
    results = []
    for path in image_paths:
        try:
            result = visualize_prediction(path, model, transform, class_names, output_dir)
            results.append(result)
            print(f"✅ {os.path.basename(path)}: {result['prediction']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"❌ {os.path.basename(path)}: {e}")
    return results
