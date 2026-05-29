from ultralytics import YOLO
import gradio as gr

# 加载你自己训练好的模型
model = YOLO("./runs/detect/final_model/weights/best.pt")

def detect_image(img):
    # 只检测行人，和训练参数保持一致
    results = model(img, classes=0, conf=0.25, imgsz=640)
    # 只返回1个结果：带框的图片
    return results[0].plot()

# 前端界面
with gr.Blocks(title="YOLOv11 行人检测系统") as demo:
    gr.Markdown("""
    # 🚀 YOLOv11 + Market-1501 行人检测系统
    ## 工程实践 / 毕业设计 —— 自主训练模型演示
    """)

    with gr.Row():
        image_in = gr.Image(type="pil", label="上传行人图片（推荐用Market-1501里的图片）")
        image_out = gr.Image(label="检测结果")

    btn = gr.Button("开始检测", variant="primary")
    btn.click(detect_image, inputs=image_in, outputs=image_out)

if __name__ == "__main__":
    demo.launch(server_port=7860)