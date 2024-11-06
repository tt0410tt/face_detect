import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 경로 설정
model_path = 'face_detect/model/training/train_face.pt'
before_test_path = 'face_detect/dataset/miniDataset'
after_test_path = 'face_detect/result/train_test/mosaic_test_ellipse'

# 모델 로드
model = YOLO(model_path)

# 원본 이미지 리스트
before_images = os.listdir(before_test_path)

# 폴더 생성
os.makedirs(after_test_path, exist_ok=True)

def apply_mosaic(image, x1, y1, x2, y2, mosaic_factor=10):
    # ROI 영역 설정
    roi = image[y1:y2, x1:x2]

    # ROI 영역의 너비와 높이를 확인하여 축소할 크기가 0 이하가 되지 않도록 함
    roi_height, roi_width = roi.shape[:2]
    if roi_width // mosaic_factor == 0 or roi_height // mosaic_factor == 0:
        # 모자이크 축소 비율이 너무 커서 적용할 수 없는 경우, 모자이크 비율을 조정
        mosaic_factor = min(roi_width, roi_height) // 2

    # ROI 영역 축소
    roi = cv2.resize(roi, (roi_width // mosaic_factor, roi_height // mosaic_factor), interpolation=cv2.INTER_LINEAR)
    # ROI 영역 확대
    roi = cv2.resize(roi, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

    # 모자이크 처리된 ROI를 원본 이미지에 적용
    image[y1:y2, x1:x2] = roi
    return image

def apply_ellipse_mosaic(image, x1, y1, x2, y2, mosaic_factor=10):
    # 타원 중심 및 크기 설정
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
    
    # 마스크 생성
    mask = np.zeros_like(image)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    
    # 타원 영역 추출
    face_region = image[y1:y2, x1:x2]
    
    # ROI 영역 축소
    roi_height, roi_width = face_region.shape[:2]
    if roi_width // mosaic_factor == 0 or roi_height // mosaic_factor == 0:
        mosaic_factor = min(roi_width, roi_height) // 2

    face_region = cv2.resize(face_region, (roi_width // mosaic_factor, roi_height // mosaic_factor), interpolation=cv2.INTER_LINEAR)
    face_region = cv2.resize(face_region, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)
    
    # 타원형 모자이크 적용
    image_copy = image.copy()
    image_copy[y1:y2, x1:x2] = face_region
    
    # 타원형 마스크 적용
    image = np.where(mask == np.array([255, 255, 255]), image_copy, image)
    return image

# 원본 파일 모자이크 처리 후 저장
for image_name in before_images:
    # 원본 이미지 경로
    src_image_path = os.path.join(before_test_path, image_name)
    print(f"Processing image: {src_image_path}")
    
    # 이미지 열기
    img_pil = Image.open(src_image_path).convert("RGB")
    img = np.array(img_pil)
    
    # OpenCV는 BGR 형식을 사용하므로 변환
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 모델로 예측
    results = model(img)
    
    # 예측 결과 모자이크 처리
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 타원형 모자이크 처리
            img = apply_ellipse_mosaic(img, x1, y1, x2, y2)

    # 결과 이미지 저장
    result_image_path = os.path.join(after_test_path, image_name)
    print(f"Saving processed image to: {result_image_path}")
    
    # 저장을 위해 다시 RGB로 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_pil.save(result_image_path)

# 결과 확인
print(f"Processed images saved to: {after_test_path}")
