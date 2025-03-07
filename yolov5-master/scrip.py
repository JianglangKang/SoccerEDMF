import cv2
import torch
from pathlib import Path
import argparse

# 导入YOLOv5模型
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

def detect_football(video_path, save_path):
    device = select_device('')

    # 加载YOLOv5模型
    model = attempt_load('yolov5s.pt', map_location=device)
    stride = int(model.stride.max())
    imgsz = 640

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # 开始读取视频
    vid = cv2.VideoCapture(video_path)
    out = None
    if save_path:
        # 如果指定了保存路径，则创建视频写入对象
        fps = vid.get(cv2.CAP_PROP_FPS)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # 转换为RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz))

        # 进行推理
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 进行推理
        pred = model(img, augment=False)[0]

        # 过滤足球类别
        pred = non_max_suppression(pred, 0.4, 0.5, classes=[0])

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # 输出足球坐标信息
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2[2] - xyxy[0]).numpy()
                    print(f"Football detected at: {xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}")

        if out:
            out.write(frame)

    vid.release()
    if out:
        out.release()

    print("Detection complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the output video")
    args = parser.parse_args()

    detect_football(args.video, args.save_path)
