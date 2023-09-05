weights_path = 'D:\GitHub\DataOnAirTeamProject\Yolov4\darknet\gingivitis\yolov4-custom_best.weights'
cfg_path = 'D:\GitHub\DataOnAirTeamProject\Yolov4\darknet\cfg\yolov4.cfg'
class_names_path = 'D:\GitHub\DataOnAirTeamProject\Yolov4\darknet\gingivitis\obj.names'
image_path = r'D:\GitHub\DataOnAirTeamProject\Yolov4\darknet\a.jpg' 



import cv2

img = cv2.imread(image_path)

with open(class_names_path, 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(224, 224), swapRB=True)

classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

classId = classIds[0]
score = scores[0]
box = boxes[0]

label = classes[classId]
confidence = score
cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=2)
text = '%s: %.2f' % (label, confidence)
cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)

cv2.imwrite('output_image.jpg', img)
