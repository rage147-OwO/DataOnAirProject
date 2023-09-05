
import base64
import cv2
import numpy as np


#DepthAI
import torch
import urllib.request
import matplotlib.pyplot as plt

model_type = "DPT_Hybrid"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False)
midas_path = r"D:\GitHub\DataOnAirTeamProject\Django\mysite\dpt_hybrid_384.pt"


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

midas.load_state_dict(torch.load(midas_path, map_location=device))

midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
output = (output / output.max()) * 255
output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
cv2.imwrite('depth_map.jpg', output)