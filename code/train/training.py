import csv
from ultralytics import YOLO
import torch
import os

class YOLOTrainer:
    def __init__(self, model_path, data_yaml, output_dir):
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.best_map = 0  
        self.best_epoch = 0  

    def log_cuda_info(self):
        # CUDA 설정 확인
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")  # True여야 GPU가 사용 가능함을 의미
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Number of GPUs: {gpu_count}")  # 사용 가능한 GPU 개수
            print(f"GPU Name: {gpu_name}")  # 첫 번째 GPU 이름

    def save_best_model(self, epoch):
        # 베스트 에포크의 모델을 저장
        best_model_dir = os.path.join(self.output_dir, 'best_epoch')
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        self.model.save(best_model_dir)
        print(f"Best model saved for epoch {epoch}")

    def train(self, epochs, batch_size, img_size):
        torch.cuda.set_device(1)
        self.model.to(torch.device('cuda:1'))
        self.log_cuda_info()
        results = self.model.train(data=self.data_yaml, epochs=epochs, batch=batch_size, imgsz=img_size)
        self.model.save(output_dir+'/train_face.pt')
        print("모델 저장")

# 학습 설정
model_path = 'face_detect/model/nomal/yolov8n-face-lindevs.pt'
data_yaml = 'face_detect/code/util/training_dataset.yaml'  # dataset.yaml 파일 경로
output_dir = 'face_detect/model/training'

torch.cuda.memory_summary()
# YOLOTrainer 객체 생성 및 학습 수행
trainer = YOLOTrainer(model_path, data_yaml, output_dir)
trainer.train(epochs=100, batch_size=8, img_size=640)
