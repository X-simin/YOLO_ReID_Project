from ultralytics import YOLO
import gradio as gr
import cv2

# 加载你训练好的 YOLOv11 模型
model = YOLO(r"E:\lenovo\Documents\工程实践\YOLO_ReID_Project\runs\detect\train-3\weights\best.pt")

def detect_with_reid(img, conf=0.25):
    # 每次检测都重新计数，确保每张图内ID从1开始
    person_id = 1
    car_id = 1

    # 检测
    results = model.predict(img, conf=conf, verbose=False)
    result = results[0]

    # 转为OpenCV格式，方便画框
    img_cv = result.orig_img.copy()

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = box.conf[0]

        if cls_name == "person":
            label = f"person:{person_id}"
            color = (0, 0, 255)  # 红色框+文字
            person_id += 1
        elif cls_name == "car":
            label = f"car:{car_id}"
            color = (0, 0, 255)  # 车辆也用红色，区分度高
            car_id += 1
        else:
            label = f"{cls_name} {conf_score:.2f}"
            color = (0, 255, 255)  # 其他类别用黄色

        # 画框
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        # 在框上方写标签
        cv2.putText(
            img_cv, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return img_cv

# 网页界面
with gr.Blocks(title="YOLOv11 目标检测与重识别系统") as demo:
    gr.Markdown("# 🎯 YOLOv11 目标检测与重识别系统（工程实践项目）")

    with gr.Row():
        in_img = gr.Image(type="pil", label="上传图片")
        out_img = gr.Image(type="numpy", label="检测结果（带ID）")

    conf_slider = gr.Slider(0.1, 1.0, 0.25, label="置信度阈值")
    btn = gr.Button("开始检测", variant="primary")

    btn.click(detect_with_reid, inputs=[in_img, conf_slider], outputs=out_img)

# 启动网页
if __name__ == '__main__':
    demo.launch(server_port=7860)
