from ultralytics import YOLO

import os
os.environ['PYTHONUTF8'] = '1'

 
import torch
print(torch.version.cuda)           # версия CUDA, если есть
print(torch.cuda.is_available())    # True = GPU доступен
print(torch.cuda.device_count())    # кол-во GPU

# Load the model.
model = YOLO('yolo11x.pt')
 
# Training.
results = model.train(
   data='dataset_yolo/dataset.yaml',
   imgsz=640,
   epochs=200,
   batch=2,
   device="0",
   name='yolov11x_200e'
)

