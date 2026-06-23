from flask import Flask, render_template, request, send_file, url_for
import os
from reid_infer import predict_image, predict_video

app = Flask(__name__)
app.secret_key = 'reid_project_2026_very_secret_key_abc123'

UPLOAD_FOLDER = './static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局变量暂存视频路径
result_video_path = None

# 首页
@app.route('/')
def index():
    return render_template("index.html")

# 图片上传接口
@app.route('/upload', methods=['POST'])
def upload_img():
    file = request.files['imgfile']
    if file and file.filename != '':
        upload_path = os.path.join(UPLOAD_FOLDER, "upload.jpg")
        file.save(upload_path)
        result_path, stat = predict_image(upload_path)
        result_url = "/" + os.path.relpath(result_path, ".").replace("\\", "/")
        return render_template("index.html", res_img=result_url, stat=stat)
    return "上传失败，请选择有效图片", 400

# 视频上传接口
@app.route('/upload_video', methods=['POST'])
def upload_vid():
    global result_video_path
    if 'vidfile' not in request.files:
        return "未检测到视频文件", 400
    vid_file = request.files['vidfile']
    if vid_file.filename == '':
        return "视频文件名为空", 400

    temp_vid = os.path.join(UPLOAD_FOLDER, "temp_video.mp4")
    vid_file.save(temp_vid)
    result_vid_path, stat = predict_video(temp_vid, save_path="./static/result.mp4")
    # 把视频路径存到全局变量
    result_video_path = result_vid_path
    return render_template("index.html", stat=stat, video_ready=True)

# 视频下载接口
@app.route('/download_video')
def download_video():
    global result_video_path
    if not result_video_path or not os.path.exists(result_video_path):
        return "视频文件不存在", 404
    return send_file(result_video_path, as_attachment=True, download_name="output_video.mp4")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
