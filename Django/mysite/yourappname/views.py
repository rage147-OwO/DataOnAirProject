import os
import cv2
import torch
import numpy as np
import base64
import trimesh
import requests
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage

CONFIG = {
    'WEIGHTS_PATH': settings.YOLOV4_WEIGHTS,
    'CFG_PATH': settings.YOLOV4_CFG,
    'CLASS_NAMES_PATH': settings.CLASS_NAMES,
    'MIDAS_PATH': settings.MIDAS_MODEL,
    'TEXTURE_PATH': settings.TEXTURE_PATH
}


def load_yolov4_model():
    net = cv2.dnn.readNetFromDarknet(CONFIG['CFG_PATH'], CONFIG['WEIGHTS_PATH'])
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(224, 224), swapRB=True)
    return model
def download_file_from_url(url, destination_path):
    """
    Downloads a file from the given URL and saves it to the provided destination path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
def load_midas_model():
    # Ensure the model exists, if not, download it.
    if not os.path.exists(CONFIG['MIDAS_PATH']):
        # The direct link to the MiDaS model on GitHub
        model_url = 'https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt'
        print("MiDaS model not found. Downloading it...")
        download_file_from_url(model_url, CONFIG['MIDAS_PATH'])
        print("Downloaded the MiDaS model successfully.")
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.load_state_dict(torch.load(CONFIG['MIDAS_PATH'], map_location=device))
    midas.to(device)
    midas.eval()
    return midas
    
def detect_and_annotate_image(img, model):
    with open(CONFIG['CLASS_NAMES_PATH'], 'r') as f:
        classes = f.read().splitlines()

    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
    classId = classIds[0]
    label = classes[classId]
    score = scores[0]
    box = boxes[0]
    confidence = score
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=2)
    text = f'{label}: {confidence:.2f}'
    cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)
    return img
    
def generate_depth_map(img, model):
    model_type = "DPT_Hybrid"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False)



    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    midas.load_state_dict(torch.load(CONFIG['MIDAS_PATH'], map_location=device))

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
    #output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('depth_map.jpg', output)
    return output


def generate_depth_to_mesh(img):
    original_textureImg = Image.open(CONFIG['TEXTURE_PATH'])
    textureImg = original_textureImg.transpose(Image.FLIP_LEFT_RIGHT)
    depth_map = np.fliplr(img)  # Flip the depth map horizontally
    depth_scale = 200.0

    # Create the vertices
    vertices = []
    uvs = []
    height, width = depth_map.shape
    for y in range(height):
        for x in range(width):
            # Normalize depth value to a reasonable range
            z = depth_scale - (depth_map[y, x] / 255.0 * depth_scale)
            vertices.append([x, height-y, z])
            u = (x / width)
            v = 1.0 - (y / height)  # Flip the v-coordinate
            uvs.append([u, v])

    # Rotate vertices around the Y axis by 180 degrees
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])  # np.pi radians is 180 degrees
    rotated_vertices = []
    for vertex in vertices:
        rotated_vertex = np.dot(rotation_matrix, [vertex[0], vertex[1], vertex[2], 1])[:3]
        rotated_vertices.append(rotated_vertex)
    vertices = np.array(rotated_vertices)

    # Create the faces by linking vertices
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            i = y * width + x
            faces.append([i, i + 1, i + width])
            faces.append([i + 1, i + width + 1, i + width])

    faces = np.array(faces)

    # Create a texture material and set the image
    material = trimesh.visual.material.SimpleMaterial(image=textureImg)

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=trimesh.visual.texture.TextureVisuals(uv=uvs, material=material))

    # Ensure the 'media' directory exists
    if not os.path.exists("media"):
        os.makedirs("media")

    # Define the path for the GLB file inside the 'media' directory
    glb_path = os.path.join("media", "output.glb")

    # Export to GLB
    mesh.export(glb_path)

    # Return the relative path to the saved GLB file
    return glb_path


@csrf_exempt
def upload_image(request):
    if request.method != 'POST':
        print('Invalid request method.')
        return JsonResponse({'message': 'Invalid request method.'})
    try:
        print(request.FILES)
        uploaded_file = request.FILES['image']
        file_bytes = uploaded_file.read()

        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        midas_model = load_midas_model()
        depth_map_img = generate_depth_map(img, midas_model)

        yolo_model = load_yolov4_model()
        detected_img = detect_and_annotate_image(img, yolo_model)
        cv2.imwrite('detected_img.jpg', detected_img)

        glb_path = generate_depth_to_mesh(depth_map_img)
        success, encoded_image = cv2.imencode('.jpg', detected_img)

        if not success:
            return JsonResponse({'message': 'EncodingError.'})

        base64_image = base64.b64encode(encoded_image).decode('utf-8')
        return JsonResponse({'message': 'ReturnImg.', 'image': base64_image, 'modelUrl': glb_path})

    except Exception as e:
        print(e)
        return JsonResponse({'message': f'Error processing the image. {str(e)}'})