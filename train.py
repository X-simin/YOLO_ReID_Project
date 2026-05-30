from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolo11n.pt")

    # 开始 GPU 训练
    model.train(
        data="voc_data.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        device=0,
        workers=0  # 最后一步稳定优化
    )