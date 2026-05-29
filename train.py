from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv11n预训练模型
    model = YOLO("yolo11n.pt")

    # 开始训练
    results = model.train(
        data="market.yaml",    # 我们刚写好的配置文件
        epochs=10,             # 训练10轮，足够学会
        imgsz=640,             # 输入图片尺寸，和标签一致
        batch=8,               # 批次大小，4060显卡完全带得动
        device=0,              # 使用GPU训练
        name="final_model",    # 模型保存的名字
        conf=0.25,             # 置信度阈值，避免漏检
        iou=0.6,               # NMS阈值，去重框
        augment=False,         # 关闭增强，避免模型学乱
        workers=4              # 数据加载线程数，加快训练
    )

    print("✅ 训练完成！模型保存在 runs/detect/final_model/weights/best.pt")