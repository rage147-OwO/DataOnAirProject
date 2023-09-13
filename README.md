## 치은염 감지를 위한 Mesh 생성 서비스


![제목 없는 다이어그램](https://github.com/rage147-OwO/DataOnAirProject/assets/96696114/4977f093-0ff0-4a4d-9b51-cf29950a5568)
[![Demo Video](http://img.youtube.com/vi/WXp8tD126k8/0.jpg)](https://www.youtube.com/watch?v=WXp8tD126k8)
[![Demo Video](http://img.youtube.com/vi/aDtGdCJB10A/0.jpg)](https://www.youtube.com/watch?v=aDtGdCJB10A)


#### 프로젝트 기간: 
2023.08 - 2023.09

#### 설명:
원격진료와 더 나은 서비스를 위해 치은염 감지와 Mesh 생성 서비스를 개발하였습니다. flutter에서 치아 이미지를 Django로 보낸 후, Object Detection을 진행 한 후, DepthMap을 생성한 뒤 gltf(Mesh)를 반환합니다.

#### 주요 기술 및 도구:
- **언어**: Python
- **프레임워크**: Django, Flutter
- **라이브러리/도구**:
  - Yolov4 (DarkNet, 객체 감지)
  - Pytorch
  - OpenCV
  - Midas (깊이 추정)
  - trimesh (깊이 맵을 3D 메쉬로 변환)

#### 설치 및 사용법:

```bash
# 레포지토리 복제
git clone [repository-link]

# 가상환경 설정 및 활성화
python -m venv venv
source venv/bin/activate

# 필요한 라이브러리 설치
pip install -r requirements.txt

# Django 서버 실행
cd Django
python manage.py runserver
```


---
### Mesh Creation Service for Gingivitis Detection


#### Project Duration: 
August 2023 - September 2023

#### Description:
Developed a service to detect gingivitis and create Mesh for improved telemedicine services. Images of teeth from Flutter are sent to Django. After object detection, a DepthMap is produced and then returned as a gltf(Mesh).

#### Main Technologies & Tools:
- **Language**: Python
- **Frameworks**: Django, Flutter
- **Libraries/Tools**:
  - Yolov4 (DarkNet, Object Detection)
  - Pytorch
  - OpenCV
  - Midas (Depth Estimation)
  - trimesh (Conversion of depth map to 3D mesh)

#### Installation & Usage:

```bash
# Clone the repository
git clone [repository-link]

# Set up and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt

# Run Django server
cd Django
python manage.py runserver
```

---
![1](https://github.com/rage147-OwO/DataOnAirProject/assets/96696114/b7ef07bb-95d6-4eb6-ab9e-2ddb89dd2aae)
![3](https://github.com/rage147-OwO/DataOnAirProject/assets/96696114/d202a3f2-f321-4631-bbda-d727c922b962)
![4](https://github.com/rage147-OwO/DataOnAirProject/assets/96696114/b4f0f7b4-7834-41ac-98b5-5755df715844)
![5](https://github.com/rage147-OwO/DataOnAirProject/assets/96696114/5f1efd21-18b0-4c1a-b3f9-bfaf4b261cbc)
![8](https://github.com/rage147-OwO/DataOnAirProject/assets/96696114/3fb9532c-e643-4dd3-9003-59337849bd7e)


