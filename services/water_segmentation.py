import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
# import matplotlib.pyplot as plt

import cv2
import numpy as np

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
device = torch.device("cpu")

model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)

for param in model.backbone.parameters():
    param.requires_grad = False

model.classifier[4] = nn.Conv2d(256, 1, kernel_size=3)

model.load_state_dict(torch.load("models/water-segmentation.pth", map_location=device))

model.eval()

# from google.colab import files
# image = files.upload()


def extract(original_image: cv2.typing.MatLike):

    original_image_pil = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL format
    original_image_pil = Image.fromarray(original_image)

    # Convert the image to tensor
    image_tensor = image_transform(original_image_pil).unsqueeze(0).to(device)

    # Perform segmentation to separate water
    outputs = model(image_tensor)['out']
    predicted_mask = torch.sigmoid(outputs).detach().cpu().squeeze().numpy()

    # Threshold the mask to obtain binary mask
    binary_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Resize binary mask to match the original image size
    binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Find contours of the water mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask from the contours
    water_contour_mask = np.zeros_like(binary_mask)
    cv2.drawContours(water_contour_mask, contours, -1, 255, cv2.FILLED)

    # Ensure water_contour_mask is of type uint8
    water_contour_mask = water_contour_mask.astype(np.uint8)

    # Create a red overlay for the water regions
    red_overlay = np.zeros_like(original_image, dtype=np.uint8)
    red_overlay[:, :, 0] = 0    # Ensure blue channel is 0
    red_overlay[:, :, 1] = 0    # Ensure green channel is 0
    red_overlay[:, :, 2] = 255  # Set red channel to maximum

    # Apply the red mask only to the water regions
    red_masked_image = cv2.bitwise_and(red_overlay, red_overlay, mask=water_contour_mask)

    # Combine the original image with the red masked image
    overlay = cv2.addWeighted(original_image, 1, red_masked_image, 0.5, 0)

    # Draw contours on the combined image
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Use the mask to extract the water region
    water_only = cv2.bitwise_and(original_image, original_image, mask=water_contour_mask)

    # Find contours based on the original image
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    original_image_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Show the results
    # plt.figure(figsize=(16, 8))
    # plt.subplot(241), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    # plt.subplot(242), plt.imshow(predicted_mask, cmap='gray'), plt.title('Predicted Mask')
    # plt.subplot(243), plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), plt.title('Overlay with Red Water')
    # plt.subplot(245), plt.imshow(binary_mask, cmap='gray'), plt.title('Binary Mask')
    # plt.subplot(246), plt.imshow(water_contour_mask, cmap='gray'), plt.title('Water Contour Mask')
    # plt.subplot(247), plt.imshow(cv2.cvtColor(water_only, cv2.COLOR_BGR2RGB)), plt.title('Extracted Water Region')
    # plt.show()

    return cv2.cvtColor(water_only, cv2.COLOR_BGR2RGB)