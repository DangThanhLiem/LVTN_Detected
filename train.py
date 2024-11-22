from ultralytics import YOLO
import torchvision.ops as ops

model = YOLO('yolov8n.pt')
def main():
    model.train(data='Dataset/SplitData/dataOffline.yaml',epochs=3, device='cpu')

if __name__ == "__main__":
    main()

