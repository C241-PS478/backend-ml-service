import torch
import torch.nn as nn
import torch.nn.functional as F
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

class DoubleConv(nn.Module):
    """Double convolution block: (convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)

model = UNet(n_channels=3, n_classes=1).to(device)

model.load_state_dict(torch.load("models/water-segmentation.pth", map_location=device))

model.eval()

# from google.colab import files
# image = files.upload()

def enlarge_and_find_best_region(mask, scale=2):
    h, w = mask.shape
    enlarged_mask = cv2.resize(mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    
    best_rect = None
    max_white_pixels = 0

    for i in range(0, h * scale - h + 1, h // 2):
        for j in range(0, w * scale - w + 1, w // 2):
            sub_mask = enlarged_mask[i:i+h, j:j+w]
            white_pixels = np.sum(sub_mask == 255)
            if white_pixels > max_white_pixels:
                max_white_pixels = white_pixels
                best_rect = (j, i, w, h)

    return best_rect, enlarged_mask

def extract(original_image: cv2.typing.MatLike):

    original_image_pil = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL format
    original_image_pil = Image.fromarray(original_image)

    # Convert the image to tensor
    image_tensor = image_transform(original_image_pil).unsqueeze(0).to(device)

    # Perform segmentation to separate water
    outputs = model(image_tensor)
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

    # Enlarge the mask and find the best region with the highest white pixel percentage
    best_rect, enlarged_mask = enlarge_and_find_best_region(water_contour_mask, scale=2)

    if best_rect:
        x, y, w, h = best_rect
        enlarged_original = cv2.resize(original_image, (original_image.shape[1] * 2, original_image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        water_only_large = enlarged_original[y:y+h, x:x+w]

        # Create a red overlay for the enlarged water regions
        red_overlay_large = np.zeros_like(water_only_large, dtype=np.uint8)
        red_overlay_large[:, :, 0] = 0    # Ensure blue channel is 0
        red_overlay_large[:, :, 1] = 0    # Ensure green channel is 0
        red_overlay_large[:, :, 2] = 255  # Set red channel to maximum

        # Apply the red mask only to the enlarged water regions
        enlarged_water_contour_mask = enlarged_mask[y:y+h, x:x+w]
        red_masked_image_large = cv2.bitwise_and(red_overlay_large, red_overlay_large, mask=enlarged_water_contour_mask)

        # Combine the enlarged original image with the red masked image
        overlay_large = cv2.addWeighted(water_only_large, 1, red_masked_image_large, 0.5, 0)

        # Extract the water region in the enlarged image
        water_only_enlarged = cv2.bitwise_and(water_only_large, water_only_large, mask=enlarged_water_contour_mask)

        # Inpaint the extracted water region in the enlarged image
        white_pixels_count = np.sum(enlarged_water_contour_mask == 255)
        total_pixels_count = enlarged_water_contour_mask.size
        coverage_percentage = (white_pixels_count / total_pixels_count) * 100

        # Inpaint the extracted water region in the enlarged image or make it full white if coverage < 25%
        if coverage_percentage < 25:
            inpainted_image_large = np.ones_like(water_only_large) * 255
        else:
            inpaint_mask_large = cv2.bitwise_not(enlarged_water_contour_mask)
            inpainted_image_large = cv2.inpaint(water_only_large, inpaint_mask_large, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # inpaint_mask_large = cv2.bitwise_not(enlarged_water_contour_mask)
        # inpainted_image_large = cv2.inpaint(water_only_large, inpaint_mask_large, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        water_only_large = None
        overlay_large = None
        enlarged_water_contour_mask = None
        water_only_enlarged = None
        inpainted_image_large = None

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

    if water_only_large is None:
        return original_image_pil, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Use the mask to extract the water region
    # water_only = cv2.bitwise_and(original_image, original_image, mask=water_contour_mask)
    
    # Show the results
    # plt.figure(figsize=(20, 10))
    # plt.subplot(331), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    # plt.subplot(332), plt.imshow(predicted_mask, cmap='gray'), plt.title('Predicted Mask')
    # plt.subplot(333), plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), plt.title('Overlay with Red Water')
    # plt.subplot(334), plt.imshow(binary_mask, cmap='gray'), plt.title('Binary Mask')
    # plt.subplot(335), plt.imshow(water_contour_mask, cmap='gray'), plt.title('Water Contour Mask')
    # plt.subplot(336), plt.imshow(cv2.cvtColor(water_only, cv2.COLOR_BGR2RGB)), plt.title('Extracted Water Region')
    # if water_only_large is not None:
    #     plt.subplot(337), plt.imshow(cv2.cvtColor(water_only_large, cv2.COLOR_BGR2RGB)), plt.title('Enlarged Extracted Water Region')
    #     plt.subplot(338), plt.imshow(cv2.cvtColor(overlay_large, cv2.COLOR_BGR2RGB)), plt.title('Overlay on Enlarged Water Region')
    #     plt.subplot(339), plt.imshow(enlarged_water_contour_mask, cmap='gray'), plt.title('Water Contour Mask on Enlarged Region')
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(cv2.cvtColor(inpainted_image_large, cv2.COLOR_BGR2RGB)), plt.title('Inpainted Extracted Water on Enlarged Region')
    # plt.show()
    
    return cv2.cvtColor(inpainted_image_large, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)