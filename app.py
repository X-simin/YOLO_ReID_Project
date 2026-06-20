import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from models.osnet import osnet_x1_0
import os
from scipy.optimize import linear_sum_assignment

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_WEIGHT_PATH = r"E:\lenovo\Documents\工程实践\reid_project\yolov8_weights\yolov8n.pt"
REID_WEIGHT = "./runs/osnet_epoch20.pth"
IMG_SIZE = (128, 256)

# 关键参数
YOLO_CONF_THRESHOLD = 0.65  
SIM_THRESHOLD = 0.7         
IOU_THRESHOLD = 0.3
MAX_MISS_FRAMES = 10       
MIN_BOX_RATIO = 1.2         
MIN_BOX_SIZE = 30          

# 自动创建文件夹
os.makedirs("./static", exist_ok=True)

# ===================== 加载模型 =====================
if not os.path.exists(YOLO_WEIGHT_PATH):
    raise FileNotFoundError(f"模型不存在，请检查：{YOLO_WEIGHT_PATH}")
yolo_model = YOLO(YOLO_WEIGHT_PATH)
print("✅ YOLO 模型加载完成")

reid_model = osnet_x1_0(num_classes=751).to(DEVICE)
reid_model.load_state_dict(torch.load(REID_WEIGHT, map_location=DEVICE))
reid_model.eval()
print("✅ ReID 模型加载完成")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 重识别 & 统计全局变量 =====================
class TrackedPerson:
    def __init__(self, feat, bbox):
        self.id = -1
        self.feat_history = [feat]
        self.bbox = bbox
        self.miss_frames = 0

    def update(self, new_feat, new_bbox):
        self.feat_history.append(new_feat)
        if len(self.feat_history) > 5:
            self.feat_history.pop(0)
        self.bbox = new_bbox
        self.miss_frames = 0

    def avg_feature(self):
        return np.mean(self.feat_history, axis=0)

tracked_persons = []
next_id = 0
frame_person_count = []
total_person_num = 0
max_frame_count = 0

def reset_all_stats():
    global tracked_persons, next_id, frame_person_count, total_person_num, max_frame_count
    tracked_persons = []
    next_id = 0
    frame_person_count = []
    total_person_num = 0
    max_frame_count = 0

def iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    xx1 = max(x1, x1_)
    yy1 = max(y1, y1_)
    xx2 = min(x2, x2_)
    yy2 = min(y2, y2_)
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (x2-x1)*(y2-y1)
    area2 = (x2_-x1_)*(y2_-y1_)
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) != 0 else 0

def filter_detections(bbox, conf, img_h, img_w):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    if conf < YOLO_CONF_THRESHOLD:
        return False
    if h / w < MIN_BOX_RATIO:
        return False
    if h < MIN_BOX_SIZE or w < MIN_BOX_SIZE:
        return False
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False
    return True

def extract_feature(crop_img):
    img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, feat = reid_model(img)
    return feat.cpu().numpy()[0]

def match_persons(detections):
    global tracked_persons, next_id
    if not tracked_persons:
        for det in detections:
            new_person = TrackedPerson(det['feat'], det['bbox'])
            new_person.id = next_id
            next_id += 1
            tracked_persons.append(new_person)
        return tracked_persons

    n_tracked = len(tracked_persons)
    n_detections = len(detections)
    cost_matrix = np.zeros((n_tracked, n_detections))

    for i, t in enumerate(tracked_persons):
        for j, d in enumerate(detections):
            sim = np.dot(t.avg_feature(), d['feat']) / (np.linalg.norm(t.avg_feature()) * np.linalg.norm(d['feat']))
            iou_score = iou(t.bbox, d['bbox'])
            cost_matrix[i, j] = -(sim * 0.8 + iou_score * 0.2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_detections = set()
    for i, j in zip(row_ind, col_ind):
        if -cost_matrix[i, j] > SIM_THRESHOLD:
            tracked_persons[i].update(detections[j]['feat'], detections[j]['bbox'])
            matched_detections.add(j)
        else:
            tracked_persons[i].miss_frames += 1

    # 未匹配的跟踪器计数
    for t in tracked_persons:
        if t.miss_frames > 0:
            t.miss_frames += 1

    # 未匹配的检测目标
    for j, det in enumerate(detections):
        if j not in matched_detections:
            new_person = TrackedPerson(det['feat'], det['bbox'])
            new_person.id = next_id
            next_id += 1
            tracked_persons.append(new_person)

    # 移除长时间未匹配的目标
    tracked_persons = [p for p in tracked_persons if p.miss_frames < MAX_MISS_FRAMES]
    return tracked_persons

def process_frame(frame):
    global max_frame_count, total_person_num
    img_h, img_w = frame.shape[:2]
    results = yolo_model(frame, classes=0, conf=YOLO_CONF_THRESHOLD)
    detections = []

    for res in results:
        boxes = res.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if not filter_detections((x1, y1, x2, y2), conf, img_h, img_w):
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                continue
            feat = extract_feature(crop)
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'feat': feat
            })

    matched_persons = match_persons(detections)
    current_frame_person = len(matched_persons)
    frame_person_count.append(current_frame_person)
    if current_frame_person > max_frame_count:
        max_frame_count = current_frame_person

    total_person_num = next_id

    for p in matched_persons:
        if p.miss_frames == 0:
            x1, y1, x2, y2 = p.bbox
            color = ((p.id * 73) % 256, (p.id * 137) % 256, (p.id * 193) % 256)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{p.id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def calc_statistics():
    if not frame_person_count:
        return {
            "total_person": 0,
            "max_frame": 0,
            "avg_frame": 0.0,
            "frame_data": []
        }
    avg = round(sum(frame_person_count) / len(frame_person_count), 2)
    return {
        "total_person": total_person_num,
        "max_frame": max_frame_count,
        "avg_frame": avg,
        "frame_data": frame_person_count
    }

def predict_image(img_path, save_path="./static/result.jpg"):
    reset_all_stats()
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("读取图片失败")
    processed_img = process_frame(img)
    cv2.imwrite(save_path, processed_img)
    stat = calc_statistics()
    return save_path, stat

def predict_video(video_path, save_path="./static/result.mp4", fps=None):
    reset_all_stats()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("读取视频失败")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = fps if fps else cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, video_fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"已处理 {frame_idx}/{total_frames} 帧")

    cap.release()
    out.release()
    print("✅ 视频处理完成")
    stat = calc_statistics()
    return save_path, stat
