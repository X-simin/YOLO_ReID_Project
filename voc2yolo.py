import os
import xml.etree.ElementTree as ET
import shutil

# 只检测这4类
classes = ["person", "car", "bus", "motorbike"]

voc_dir = "VOC2012"
img_dir = os.path.join(voc_dir, "JPEGImages")
ann_dir = os.path.join(voc_dir, "Annotations")

out_train_img = "datasets/images/train"
out_val_img = "datasets/images/val"
out_train_label = "datasets/labels/train"
out_val_label = "datasets/labels/val"

os.makedirs(out_train_img, exist_ok=True)
os.makedirs(out_val_img, exist_ok=True)
os.makedirs(out_train_label, exist_ok=True)
os.makedirs(out_val_label, exist_ok=True)

xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
split = int(len(xml_files) * 0.8)
train_xmls = xml_files[:split]
val_xmls = xml_files[split:]

def process(xml_name, out_label_dir):
    tree = ET.parse(os.path.join(ann_dir, xml_name))
    root = tree.getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        bnd = obj.find("bndbox")
        x1 = float(bnd.find("xmin").text)
        y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text)
        y2 = float(bnd.find("ymax").text)

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    img_name = xml_name.replace(".xml", ".jpg")
    txt_name = xml_name.replace(".xml", ".txt")
    with open(os.path.join(out_label_dir, txt_name), "w") as f:
        f.write("\n".join(lines))
    return img_name

for f in train_xmls:
    img = process(f, out_train_label)
    shutil.copy(os.path.join(img_dir, img), out_train_img)
for f in val_xmls:
    img = process(f, out_val_label)
    shutil.copy(os.path.join(img_dir, img), out_val_img)

print("✅ 数据集转换完成！")