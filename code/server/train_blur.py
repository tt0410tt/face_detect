import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

def tarin_blur(file_name):
    # 모델 로드
    model_path = 'face_detect/model/training/train_face.pt'
    model = YOLO(model_path)

    # 비디오 파일 로드
    video_path = 'face_detect/code/server/static/upload/'+file_name
    cap = cv2.VideoCapture(video_path)

    # 비디오 작성기 설정
    output_path = 'face_detect/code/server/static/download/blur_'+file_name
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
                
                # 사각형 영역 추출 및 블러 처리
                face_region = img[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                
                # 블러 처리된 영역을 원본 이미지에 붙여넣기
                img[y1:y2, x1:x2] = blurred_face
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
