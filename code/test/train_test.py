import os
import shutil
import random
from ultralytics import YOLO
from PIL import Image, ImageDraw

# 경로 설정
val_images_path = 'face_detect/dataset/miniDataset'
after_test_path = 'face_detect/result/train_test/AI_model_test'
model_path = 'face_detect/model/training/train_face.pt'
#model_path = 'runs/detect/train/weights/best.pt'

# 모델 로드
model = YOLO(model_path)

# 10장 뽑기
val_images = os.listdir(val_images_path)

# 원본 파일 복사 및 안면 인식 후 파일 저장
for image_name in val_images:
    # 이미지 열기
    print(val_images_path+'/'+image_name)
    img = Image.open(val_images_path+'/'+image_name).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # 모델로 예측
    results = model(img)
    
    # 예측 결과 그리기
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            label = f"face {float(box.conf):.2f}"
            draw.text((x1, y1), label, fill="blue")

    # 결과 이미지 저장
    result_image_path = os.path.join(after_test_path, image_name)
    img.save(result_image_path)


print("완료했습니다.")