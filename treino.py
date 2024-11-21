import os
from ultralytics import YOLO

def treino(data_yaml, epochs=100, img_size=640, model='yolo11n.pt', patience = 10):
    print(f"Iniciando o treinamento do YOLOv8 com {epochs} épocas e tamanho de imagem {img_size}...")
    
    model = YOLO(model) 
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size, patience=patience)

if __name__ == "__main__":
    data_yaml = 'data.yaml' 
    treino(data_yaml, epochs=100, img_size=640, model='yolo11n.pt',patience=10)
