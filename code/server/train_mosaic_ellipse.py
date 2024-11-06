import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

def apply_mosaic(image, x1, y1, x2, y2, mosaic_factor=50):
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

def mosaic_ellipse(file_name, mosaic_factor=50):
    # 모델 로드
    model_path = 'face_detect/model/training/train_face.pt'
    model = YOLO(model_path)

    # 비디오 파일 로드
    video_path = 'face_detect/code/server/static/upload/' + file_name
    cap = cv2.VideoCapture(video_path)

    # 비디오 작성기 설정
    output_path = 'face_detect/code/server/static/download/mosaic_ellipse_' + file_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 모델로 예측
        results = model(img_pil)

        # 예측 결과 그리기
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()

                # 타원 중심 및 크기 설정
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = ((x2 - x1) // 2, (y2 - y1) // 2)

                # 마스크 생성
                mask = np.zeros_like(img)
                cv2.ellipse(mask, center, axes, 0, 0, 360, (1, 1, 1), -1)

                # 원본 이미지 복사
                img_copy = img.copy()

                # 모자이크 처리
                img_copy = apply_mosaic(img_copy, x1, y1, x2, y2, mosaic_factor)

                # 타원형 마스크 적용
                img = np.where(mask == np.array([1, 1, 1]), img_copy, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 프레임을 비디오에 작성
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # 진행 상태 출력
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.2f}%")

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved to: {output_path}")

