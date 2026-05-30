from ultralytics import YOLO
import gradio as gr

# Windows 必须加这个
if __name__ == '__main__':
    # 加载你训练好的 YOLOv11 模型
    model = YOLO(r"E:\lenovo\Documents\工程实践\YOLO_ReID_Project\runs\detect\train-3\weights\best.pt")


    # 检测函数
    def detect(img, conf=0.25):
        results = model.predict(img, conf=conf, device=0)
        return results[0].plot()


    # 网页界面
    with gr.Blocks(title="YOLOv11 行人检测系统") as demo:
        gr.Markdown("# 🎯 YOLOv11 行人检测系统（工程实践项目）")

        with gr.Row():
            in_img = gr.Image(type="pil", label="上传图片")
            out_img = gr.Image(type="pil", label="检测结果")

        conf_slider = gr.Slider(0.1, 1.0, 0.25, label="置信度阈值")
        btn = gr.Button("开始检测", variant="primary")

        btn.click(detect, inputs=[in_img, conf_slider], outputs=out_img)

    # 启动网页
    demo.launch(server_port=7860)