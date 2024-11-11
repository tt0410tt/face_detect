import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
def apply_mosaic(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, mosaic_factor: int = 10) -> np.ndarray:
    """
    지정된 영역에 모자이크를 적용합니다.

    Args:
        image (np.ndarray): 원본 이미지 배열
        x1 (int): 모자이크를 적용할 영역의 왼쪽 상단 x 좌표
        y1 (int): 모자이크를 적용할 영역의 왼쪽 상단 y 좌표
        x2 (int): 모자이크를 적용할 영역의 오른쪽 하단 x 좌표
        y2 (int): 모자이크를 적용할 영역의 오른쪽 하단 y 좌표
        mosaic_factor (int): 모자이크 강도 (기본값: 10)

    Returns:
        np.ndarray: 모자이크가 적용된 이미지 배열
    """
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

def mosaic(file_name: str) -> None:
    """
    비디오에서 얼굴을 감지하고 얼굴 영역을 모자이크 처리한 후,
    처리된 비디오를 저장합니다.

    Args:
        file_name (str): 처리할 비디오 파일의 이름

    Returns:
        'face_detect/code/server/static/download/mosaic_' + file_name
        에 파일이 저장된다.
    """
    current_file = Path(__file__).resolve()
    main_folder = current_file.parent.parent.parent
    # 모델 로드
    model_path = main_folder / "model" / "training" / "train_face.pt"
    model = YOLO(model_path)

    # 비디오 파일 로드
    video_path = main_folder / "code" / "server" / "static" / "upload" / file_name
    cap = cv2.VideoCapture(video_path)

    # 비디오 작성기 설정
    file_name1="mosaic_"+file_name
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
                # 모자이크 처리
                img = apply_mosaic(img, x1, y1, x2, y2)
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
