import numpy as np
import cv2
import trimesh

# Read the depth map
depth_map_image_path = r'D:\GitHub\DataOnAirTeamProject\DepthMapToObj\image.png'
depth_map =  cv2.imread(depth_map_image_path)
depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
depth_scale = 100.0
# Create the vertices
vertices = []
uvs = []
height, width = depth_map.shape
for y in range(height):
    for x in range(width):
        # Normalize depth value to a reasonable range
        z = depth_map[y, x] / 255.0 * depth_scale
        vertices.append([x, y, z])
        uvs.append([x / width, y / height])


vertices = np.array(vertices)

# Create the faces by linking vertices
faces = []
for y in range(height - 1):
    for x in range(width - 1):
        # Define vertices for the face
        i = y * width + x
        faces.append([i, i + 1, i + width])
        faces.append([i + 1, i + width + 1, i + width])

faces = np.array(faces)

# Create a mesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces,visual=trimesh.visual.texture.TextureVisuals(uv=uvs))

# Export to GLTF
mesh.export('output.glb')

