from ultralytics import YOLO
import gradio as gr
import cv2
import numpy as np

model = YOLO(r"E:\lenovo\Documents\工程实践\YOLO_ReID_Project\runs\detect\train-3\weights\best.pt")

def detect_with_reid(img, conf=0.25):
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = model.predict(img_cv, conf=conf, iou=0.45, classes=[0], verbose=False)
    result = results[0]

    person_id = 1
    person_cnt = 0

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name == "person":
            label = f"person:{person_id}"
            color = (0, 0, 255)  
            person_id += 1
            person_cnt += 1

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_cv, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    info = f"检测到人物总数：{person_cnt}"

    img_final = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_final, info

with gr.Blocks(
    title="YOLO 人物检测",
    theme=gr.themes.Soft(),
    css="""
    #title {text-align: center; font-size: 2em; font-weight: bold; margin-bottom: 1rem;}
    .gradio-container {padding: 1.5rem; max-width: 1200px; margin: 0 auto;}
    """
) as demo:
    gr.Markdown("# 🎯 YOLO 人物检测系统", elem_id="title")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 上传图片")
            in_img = gr.Image(type="pil", label="")

        with gr.Column():
            gr.Markdown("### 📷 检测结果（带ID）")
            out_img = gr.Image(type="numpy", label="")

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=3):
            conf_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                label="置信度阈值", info="推荐默认值 0.25，可根据效果微调"
            )
            info_text = gr.Textbox(label="📊 统计结果", interactive=False)

        with gr.Column(scale=1):
            btn = gr.Button("开始检测", variant="primary", size="lg")

    btn.click(
        detect_with_reid,
        inputs=[in_img, conf_slider],
        outputs=[out_img, info_text]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
