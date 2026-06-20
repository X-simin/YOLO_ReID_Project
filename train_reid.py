import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.osnet import osnet_x1_0

# ===================== 配置参数 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./datasets/Market-1501-v15.09.15/bounding_box_train"
BATCH_SIZE = 16
EPOCHS = 20
IMG_SIZE = (128, 256)
NUM_CLASSES = 751  # Market1501 固定751个行人ID

# ===================== 数据集加载 =====================
class MarketTrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = []
        self.labels = []
        label_set = set()

        for img_name in os.listdir(root):
            if img_name.endswith(('.jpg', '.png')):
                # Market1501 文件名规则: 第一个数字为行人ID
                pid = int(img_name.split('_')[0])
                self.imgs.append(os.path.join(root, img_name))
                self.labels.append(pid)
                label_set.add(pid)
        self.label_map = {pid: idx for idx, pid in enumerate(sorted(label_set))}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.label_map[self.labels[idx]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 图像预处理
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集&迭代器
train_dataset = MarketTrainDataset(DATA_ROOT, train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ===================== 模型、损失、优化器 =====================
model = osnet_x1_0(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ===================== 训练流程 =====================
if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    print(f"训练集数量: {len(train_dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs, _ = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

        # 每5轮保存权重
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"./runs/osnet_epoch{epoch+1}.pth")
            print(f"模型权重已保存: runs/osnet_epoch{epoch+1}.pth")

    print("训练完成！")