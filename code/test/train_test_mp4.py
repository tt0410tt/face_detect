import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 모델 로드
model_path = 'face_detect/model/training/train_face.pt'
model = YOLO(model_path)

# 비디오 파일 로드
video_path = 'face_detect/code/server/static/upload/chating.mp4'
cap = cv2.VideoCapture(video_path)

# 비디오 작성기 설정
output_path = 'face_detect/code/server/static/download/chating.mp4'
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
    
    # 모델로 예측
    results = model(img_pil)
    
    # 예측 결과 그리기
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            
            # 감지된 객체에 사각형 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Conf: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 프레임을 비디오에 작성
    out.write(frame)
    
    # 진행 상태 출력
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    print(f"Progress: {progress:.2f}%")
    
    # 프레임 보여주기 (원한다면)
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_path}")
