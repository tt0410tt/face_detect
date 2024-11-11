import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
def train_blur_ellipse(file_name: str) -> None:
    """
    비디오에서 얼굴을 감지하고 타원형 마스크를 사용하여 얼굴 영역을 블러 처리한 후,
    처리된 비디오를 저장합니다.

    Args:
        file_name (str): 처리할 비디오 파일의 이름

    Returns:
        code/server/static/download/blur_ellipse_/"filename.np4"
    """
    current_file = Path(__file__).resolve()
    main_folder = current_file.parent.parent.parent
    # 모델 로드
    model_path = main_folder / "model" / "training" / "train_face.pt"
    model = YOLO(str(model_path))

    # 비디오 파일 로드
    video_path = main_folder / "code" / "server" / "static" / "upload" / file_name
    cap = cv2.VideoCapture(video_path)

    # 비디오 작성기 설정
    file_name1="blur_ellipse_"+file_name
    output_path = main_folder / "code" / "server" / "static" / "download"/ file_name1
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
                cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
                
                # 원본 이미지 복사
                img_copy = img.copy()

                # 사각형 영역 추출 및 블러 처리
                face_region = img[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)

                # 블러 처리된 영역을 원본 이미지에 붙여넣기
                img_copy[y1:y2, x1:x2] = blurred_face
                
                # 타원형 마스크 적용
                img = np.where(mask == np.array([255, 255, 255]), img_copy, img)

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
