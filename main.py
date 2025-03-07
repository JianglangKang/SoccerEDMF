from collections import defaultdict
import time
import os
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import torch
'''球员分类'''
from inference.hsv_classifier import HSVClassifier
from inference.filters import filters

# ------todo 修改点   透视变换
src_target_points = np.array([(77, 460), (300, 1075), (115, 215), (700, 1030)],
                                 dtype=np.float32)  # 24_01 2 3 4
dest_target__points = np.array([(588, 400), (588, 715), (638, 23), (638, 778)], dtype=np.float32)
src_twoD_points = np.array([(115, 215), (700, 1030), (964, 80), (1760, 285)], dtype=np.float32)
dest_twoD_points = np.array([(638, 23), (638, 778), (1220, 23), (1220, 624)], dtype=np.float32)
position_middle = [(115, 215), (700, 1030)]
H = cv2.getPerspectiveTransform(src_target_points, dest_target__points)
H0 = cv2.getPerspectiveTransform(src_twoD_points, dest_twoD_points)


class ColorCounter:
    def __init__(self):
        self.color_counts = defaultdict(int)

    def add_color(self, color):
        self.color_counts[color] += 1

    def get_most_frequent_color(self):
        if not self.color_counts:
            return None
        most_frequent_color = max(self.color_counts, key=self.color_counts.get)
        return most_frequent_color, self.color_counts[most_frequent_color]
# 对球衣颜色进行计数，找出最多次出现的球衣颜色，给二维图中颜色进行赋值
counters = defaultdict(ColorCounter)

color={}
'''球员处理'''
classfier = HSVClassifier(filters=filters)

# todo 修改添加的球衣颜色
def color_predict(img,pos):
    img_patch = img[max(int(pos[1]), 0):max(int(pos[3]), 0),
                max(int(pos[0]), 0):max(int(pos[2]), 0)]

    height, width, _ = img_patch.shape

    if height != 0 and width != 0:
        result = classfier.predict_img(img_patch)
    else:
        result = 'other'

    if result == 'team_red':
        return [0, (0, 0, 255)]
    elif result == 'team_white':
        return [1, (255, 255, 255)]
    elif result == 'team_blue':
        return [2, (233, 180, 90)]
    elif result == 'team_black':
        return [3, (0, 0, 0)]
    # elif result == 'Argentina':
    #     return [0, (255, 255, 0)]  # 135 206 250
    # elif result == 'France':
    #     return [1, (205, 0, 0)]
    # elif result == 'Argentina_goalie':
    #     return [0, (0, 255, 0)]
    # elif result == 'France_goalie':
    #     return [1, (0, 255, 255)]
    # elif result == 'Referee':
    #     return [2, (0, 0, 255)]
    elif result == 'other':
        return [3, (0, 255, 0)]

'''kjl'''
def Po1(socceres, position_middle):
    queue = []

    for soccer in socceres:
        n = 0
        x = round(soccer[0])
        y = round(soccer[1])
        w = round(soccer[2])
        h = round(soccer[3])
        cross_product=(position_middle[1][0] - position_middle[0][0]) * (y - position_middle[0][1]) - (x -position_middle[0][0]) * (position_middle[1][1] - position_middle[0][1])
        if cross_product >= 0: #左侧或在中线上
            # y = round(y + h / 2)
            queue.append([n, x, y])

    return queue

def Po2(socceres, position_middle):
    queue = []

    for soccer in socceres:
        n = 0
        x = round(soccer[0])
        y = round(soccer[1])
        w = round(soccer[2])
        h = round(soccer[3])
        cross_product=(position_middle[1][0] - position_middle[0][0]) * (y - position_middle[0][1]) - (x -position_middle[0][0]) * (position_middle[1][1] - position_middle[0][1])
        if cross_product < 0: #左侧或在中线上
            # y = round(y + h / 2)
            queue.append([n, x, y])
    return queue

def Po3(persons, position_middle):
    queue = []

    for person in persons:
        n = round(person[0])
        x = round(person[1])
        y = round(person[2])
        w = round(person[3])
        h = round(person[4])
        cross_product=(position_middle[1][0] - position_middle[0][0]) * (y - position_middle[0][1]) - (x -position_middle[0][0]) * (position_middle[1][1] - position_middle[0][1])
        if cross_product >= 0: #左侧或在中线上
            y = round(y + h / 2)
            queue.append([n, x, y])

    return queue

def Po4(persons, position_middle):
    queue = []

    for person in persons:
        n = round(person[0])
        x = round(person[1])
        y = round(person[2])
        w = round(person[3])
        h = round(person[4])
        cross_product=(position_middle[1][0] - position_middle[0][0]) * (y - position_middle[0][1]) - (x -position_middle[0][0]) * (position_middle[1][1] - position_middle[0][1])
        if cross_product < 0: #左侧或在中线上
            y = round(y + h / 2)
            queue.append([n, x, y])
    return queue

def get_pos(video_path):
    cap = cv2.VideoCapture(video_path)
    pos_soccers=[]
    fra_cnt = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            im_height, im_width, _ = frame.shape
            model_soccer.conf = 0.65
            # 将帧从 BGR 转换为 RGB 进行检测
            img_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model_soccer(img_cvt)
            soccerpos_dic = {}
            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                # 计算 xywh 坐标
                w = x2 - x1  # 宽度
                h = y2 - y1  # 高度
                x_center = x1 + w // 2  # 中心点 x
                y_center = y1 + h // 2  # 中心点 y
                soccerpos_dic.setdefault(fra_cnt, []).append([x_center,y_center,w,h])

            pos_soccers.append(soccerpos_dic)
            fra_cnt += 1
        else:
            break
    cap.release()
    return pos_soccers

def show_soccer(soccer_persons_video_path, pos_soccers):
    cap = cv2.VideoCapture(soccer_persons_video_path)
    fps= int(cap.get(cv2.CAP_PROP_FPS))
    temp_output_path='outputs/temp_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))  # Overwrite the original video

    fra_cnt = 0
    soccers_pos=[]
    # 存储追踪历史
    track_history = defaultdict(lambda: [])

    '''实验'''
    # 获取足球在哪些帧出现过
    keys_list = [list(d.keys()) for d in pos_soccers]
    keys = [i for i, sublist in enumerate(keys_list) if sublist]
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            soccer_dic = {}
            socceres = []
            '''实验'''
            # 确保每一帧都有track
            track = track_history[0]
            if fra_cnt in keys:
                for pos_soccer in list(pos_soccers[fra_cnt].values())[0]:
                    socceres.append(pos_soccer)
                    track.append((pos_soccer[0], pos_soccer[1]))  # x, y中心点
            # print(track)
            if len(track) > 10:  # 检查轨迹的长度是否超过30个时间步长（len(track) > 10）。如果是，则从轨迹的开头弹出一个元素，以确保最新的10个位置信息保留在轨迹中（track.pop(0)）。
                track.pop(0)
                # todo 绘制追踪轨迹
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
            if len(socceres) != 0:
                # 得到转换2d的多个足球的位置 xywh

                # ------todo 修改
                # 区分在左半场还是右半场
                position_left = Po1(socceres, position_middle)
                position_right = Po2(socceres, position_middle)
                # print('--')

                # 透视变换获得新的坐标
                for i in range(len(position_left)):
                    points = [position_left[i][1], position_left[i][2]]
                    # 在导入的视频中添加id
                    # 在导入的视频中画圈
                    cv2.circle(frame, (position_left[i][1], position_left[i][2]), 6, (255, 0, 0), -1)  # Blue
                    x1 = int((points[0] * H[0][0] + points[1] * H[0][1] + H[0][2]) / (
                            points[0] * H[2][0] + points[1] * H[2][1] + H[2][2])) - 10
                    y1 = int((points[0] * H[1][0] + points[1] * H[1][1] + H[1][2]) / (
                            points[0] * H[2][0] + points[1] * H[2][1] + H[2][2]))
                    soccer_dic.setdefault(fra_cnt, []).append((x1, y1))

                for i in range(len(position_right)):
                    points = [position_right[i][1], position_right[i][2]]

                    cv2.circle(frame, (position_right[i][1], position_right[i][2]), 6, (255, 0, 0),-1)  # Blue
                    x1 = int((points[0] * H0[0][0] + points[1] * H0[0][1] + H0[0][2]) / (
                            points[0] * H0[2][0] + points[1] * H0[2][1] + H0[2][2])) - 10
                    y1 = int((points[0] * H0[1][0] + points[1] * H0[1][1] + H0[1][2]) / (
                            points[0] * H0[2][0] + points[1] * H0[2][1] + H0[2][2]))
                    soccer_dic.setdefault(fra_cnt, []).append((x1, y1))
            # 获取球二维坐标
            soccers_pos.append(soccer_dic)
            # 获取球员二维坐标

            # print(soccers_pos)
            out.write(frame)
            fra_cnt += 1
        else:
            break

    cap.release()
    out.release()
    os.replace(temp_output_path, soccer_persons_video_path)
    return soccers_pos

def show_persons(video_path,soccer_persons_video_path):

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 设置视频编码器，并创建输出视频对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(soccer_persons_video_path, fourcc, fps, (width, height))
    pos_persons = []
    # 获取二维坐标
    persons_pos = []
    pos_speed={}
    pos_speed_all = {}
    fra_cnt = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            im_height, im_width, _ = frame.shape
            person_dic = {}
            persons = []
            # 当检测到球员
            result = model_person.track(frame, persist=True, tracker="botsort.yaml")
            for boxes in result[0].cpu().numpy().boxes:
                for box_person in boxes.xywh:
                    # 获取球员id+真实坐标
                    ID=boxes.id[0]

                    persons.append((int(boxes.id[0]), box_person[0], box_person[1], box_person[2], box_person[3]))
                    pos_persons.append(persons)
                    xyxy = boxes.xyxy[0].astype(dtype=int).tolist()
                    class_result = color_predict(frame, xyxy)
                    counters[str(int(ID))].add_color(class_result[1])
                    color[str(int(boxes.id[0]))] = class_result[1]


            # 球员可视化效果
            if len(persons) != 0:
                # 转二维图坐标 id x y
                position_left = Po3(persons, position_middle)
                position_right = Po4(persons, position_middle)
                # 透视变换获得新的坐标
                for i in range(len(position_left)):
                    points = [position_left[i][1], position_left[i][2]]
                    label = '{}{:d}'.format("", position_left[i][0])
                    # 在导入的视频中添加id

                    x1 = int((points[0] * H[0][0] + points[1] * H[0][1] + H[0][2]) / (
                            points[0] * H[2][0] + points[1] * H[2][1] + H[2][2])) - 10
                    y1 = int((points[0] * H[1][0] + points[1] * H[1][1] + H[1][2]) / (
                            points[0] * H[2][0] + points[1] * H[2][1] + H[2][2]))
                    person_dic.setdefault(fra_cnt, []).append((label, x1, y1))

                    if label in pos_speed:
                        pos_speed_all[label].append((x1,y1))
                        if (fra_cnt % 5 == 0):
                            pos_speed[label].append((x1,y1))
                    else:
                        pos_speed_all.setdefault(label, []).append((x1, y1))
                        pos_speed.setdefault(label, []).append((x1, y1))

                for i in range(len(position_right)):
                    points = [position_right[i][1], position_right[i][2]]
                    label = '{}{:d}'.format("", position_right[i][0])

                    x1 = int((points[0] * H0[0][0] + points[1] * H0[0][1] + H0[0][2]) / (
                            points[0] * H0[2][0] + points[1] * H0[2][1] + H0[2][2])) - 10
                    y1 = int((points[0] * H0[1][0] + points[1] * H0[1][1] + H0[1][2]) / (
                            points[0] * H0[2][0] + points[1] * H0[2][1] + H0[2][2]))
                    person_dic.setdefault(fra_cnt, []).append((label, x1, y1))
                    if label in pos_speed:
                        pos_speed_all[label].append((x1, y1))
                        if (fra_cnt % 5 == 0):
                            pos_speed[label].append((x1,y1))
                    else:
                        pos_speed_all.setdefault(label, []).append((x1, y1))
                        pos_speed.setdefault(label, []).append((x1, y1))
                '''end'''

                # print("-----")
            for ID, x, y, w, h in persons:
                # print(ID, x, y, w, h)
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                # 在视频中画出球员框和id和球衣颜色

                cv2.rectangle(frame, (x1, y1), (x2, y2), color=counters[str(ID)].get_most_frequent_color()[0], thickness=2)

                # cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0,0,0), thickness=2)
                # print(str[ID])
                caption =f"{ID}"
                w1, h1 = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]

                cv2.rectangle(frame, (x1 - 3, y1 - 25), (x1 + w1 + 5, y1), counters[str(ID)].get_most_frequent_color()[0], -1)

                cv2.putText(frame, str(ID), (int(x1), int(y1) - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8,
                            color=(128, 128, 128), thickness=2)

            # 获取球员二维坐标
            persons_pos.append(person_dic)
            out.write(frame)
            fra_cnt += 1
        else:
            break

    # Release the video capture object and close the output video
    print(counters)
    cap.release()
    out.release()
    return persons_pos,pos_speed_all
import math

import math

# todo window_size滑动窗口大小（几帧求一个速度） field_length真实球场宽度  field_2d_length给出的二维图像素宽度
def calculate_speed_distance(positions, window_size=5, field_length=105, field_2d_length=1170):
    speeds = [0] * len(positions)  # 初始化速度列表，初始值为0
    speed_max = 0
    speed_count = [0, 0, 0, 0, 0, 0]
    dis_sum = 0.0

    zhuang = field_length / field_2d_length  # 实际场地长度与2D场地长度的比例

    i = 0
    while i + window_size <= len(positions):
        # 计算窗口内的速度
        x_prev, y_prev = positions[i]
        x_next, y_next = positions[i + window_size - 1]
        distance = math.sqrt((x_next - x_prev) ** 2 + (y_next - y_prev) ** 2) * zhuang
        dis_sum += distance
        speed = (distance * fps) / window_size  # 计算速度
        speed = round(speed, 2)
        if speed > speed_max:
            speed_max = speed
        # 判断速度等级次数计算在此速度范围内跑动时间
        if speed >= 0 and speed <= 1.5:
            speed_count[0] += window_size
        elif speed <= 2.5:
            speed_count[1] += window_size
        elif speed <= 3.5:
            speed_count[2] += window_size
        elif speed <= 4.5:
            speed_count[3] += window_size
        elif speed <= 5.5:
            speed_count[4] += window_size
        elif speed > 6.5:
            speed_count[5] += window_size


        # 将该速度赋值给当前窗口内的所有帧
        for j in range(window_size):
            speeds[i + j] = speed

        i += window_size

    # 处理最后不足窗口大小的帧，单独计算
    if i < len(positions):
        x_prev, y_prev = positions[i]
        x_next, y_next = positions[-1]
        distance = math.sqrt((x_next - x_prev) ** 2 + (y_next - y_prev) ** 2) * zhuang
        dis_sum += distance
        remain_frame = (len(positions) - i)
        speed = (distance * fps) / remain_frame  # 计算速度
        speed = round(speed, 2)
        if speed > speed_max:
            speed_max = speed

        # 判断速度等级次数计算在此速度范围内跑动时间
        if speed >= 0 and speed <= 3.24:
            speed_count[0] += remain_frame
        elif speed <= 3.92:
            speed_count[1] += remain_frame
        elif speed <= 4.75:
            speed_count[2] += remain_frame
        elif speed <= 5.86:
            speed_count[3] += remain_frame
        elif speed <= 6.85:
            speed_count[4] += remain_frame
        elif speed > 6.85:
            speed_count[5] += remain_frame


        # 将该速度赋值给剩余的帧
        for j in range(i, len(positions)):
            speeds[j] = speed

    # 将速度等级次数转化为时间
    for i in range(0, 6):
        speed_count[i] = round(speed_count[i] / fps, 2)

    dis_sum = round(dis_sum, 2)
    speed_max = round(speed_max, 2)
    speed_avg = round(dis_sum * fps / len(positions), 2)
    return speeds, speed_max, speed_avg, speed_count, dis_sum



# 滑动平均窗口大小
window_size = 5
# 初始化一个空的列表，用于存储每个目标的位置历史记录
position_history = {}
'''位置平滑处理'''
def smooth_position(label, x, y):
    # 如果是第一次出现的目标，则直接返回当前位置
    if label not in position_history:
        position_history[label] = [(x, y)]
        return x, y

    # 获取该目标的位置历史记录
    history = position_history[label]

    # 将当前位置添加到历史记录中
    history.append((x, y))

    # 如果历史记录长度超过窗口大小，则移除最早的位置记录
    if len(history) > window_size:
        history.pop(0)

    # 计算窗口内位置的平均值
    smoothed_x = np.mean([pos[0] for pos in history])
    smoothed_y = np.mean([pos[1] for pos in history])

    return int(smoothed_x), int(smoothed_y)

'''二维图'''
def soccer_persons_2D(soccers_pos,persons_pos,img_dir,excel_path,video_path,soccer_persons_2D_path,pos_speed_all):
# def soccer_persons_2D( persons_pos, img_dir, excel_path, video_path, soccer_persons_2D_path,pos_speed_all):
    speed_dic = {}
    speed_max_dic = {}
    speed_avg_dic = {}
    speed_count_dic = {}
    dis_sum_dic = {}

    # 计算速度
    for key, values in pos_speed_all.items():
        # print(values)
        speeds, speed_max, speed_avg, speed_count, dis_sum = calculate_speed_distance(values,8)
        if key in speed_dic:
            speed_dic[key].append(speeds)
        else:
            speed_dic.setdefault(key, []).append(speeds)
            speed_max_dic.setdefault(key, speed_max)
            speed_avg_dic.setdefault(key, speed_avg)
            speed_count_dic.setdefault(key, speed_count)
            dis_sum_dic.setdefault(key, dis_sum)

    data = []
    for player_id in counters:
        most_frequent_color, count = counters[player_id].get_most_frequent_color()
        data.append([player_id, most_frequent_color, count])
    #将id出现最多颜色进行保存excel用于后续分队
    ids, colors, counts = zip(*data)
    print(counters)
    # 将数据存储在excel表格中
    max_data = sorted((int(k), v) for k, v in speed_max_dic.items())
    max_keys, max_values = zip(*max_data)
    avg_data = sorted((int(k), v) for k, v in speed_avg_dic.items())
    avg_keys, avg_values = zip(*avg_data)
    # 将字典的键转换为整数类型，然后按照整数大小排序
    sorted_data = sorted((int(k), v) for k, v in dis_sum_dic.items())
    # 分离排序后的键和值
    sorted_keys, sorted_values = zip(*sorted_data)
    # 速度等级导出
    sorted_0 = sorted(speed_count_dic.keys(), key=lambda x: int(x))
    level_0 = [speed_count_dic[key][0] for key in sorted_0]
    level_1 = [speed_count_dic[key][1] for key in sorted_0]
    level_2 = [speed_count_dic[key][2] for key in sorted_0]
    level_3 = [speed_count_dic[key][3] for key in sorted_0]
    level_4 = [speed_count_dic[key][4] for key in sorted_0]
    level_5 = [speed_count_dic[key][5] for key in sorted_0]

    sr0 = pd.Series(list(colors), index=sorted_keys)
    sr1 = pd.Series(list(sorted_values), index=sorted_keys)
    sr2 = pd.Series(list(avg_values), index=sorted_keys)
    sr3 = pd.Series(list(max_values), index=sorted_keys)
    sr_0 = pd.Series(level_0, index=sorted_keys)
    sr_1 = pd.Series(level_1, index=sorted_keys)
    sr_2 = pd.Series(level_2, index=sorted_keys)
    sr_3 = pd.Series(level_3, index=sorted_keys)
    sr_4 = pd.Series(level_4, index=sorted_keys)
    sr_5 = pd.Series(level_5, index=sorted_keys)

    # 创建DataFrame
    df = pd.DataFrame({'Distance': sr1, 'Speed_avg': sr2, 'Speed_max': sr3,
                       'Speed_Lv0': sr_0, 'Speed_Lv1': sr_1, 'Speed_Lv2': sr_2,
                       'Speed_Lv3': sr_3, 'Speed_Lv4': sr_4, 'Speed_Lv5': sr_5, 'Color': sr0})
    df = df.T
    df.to_excel(excel_path, index=True, sheet_name="0-15")

    speed_cnt = {}
    for ID in speed_dic.keys():
        speed_cnt[ID] = 0
    # print(speed_cnt)

    cap=cv2.VideoCapture(video_path)
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    frame_cnt=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    videoWriter = cv2.VideoWriter(soccer_persons_2D_path, fourcc, fps, (template_w,template_h))

    # 获取球所在帧号（列表）
    soccer_list = [list(d.keys()) for d in soccers_pos]
    soccer_keys = [i for i, sublist in enumerate(soccer_list) if sublist]
    # 获取球员所在帧号
    persons_list = [list(d.keys()) for d in persons_pos]
    persons_keys = [i for i, sublist in enumerate(persons_list) if sublist]
    for i in range(frame_cnt):
        model_image = cv2.imread(img_dir)
        arc = cv2.resize(model_image, (template_w, template_h))
        # 球在2D可视化
        if i in soccer_keys:
            for soccer_pos in list(soccers_pos[i].values())[0]:
                cv2.circle(arc, soccer_pos, 6, (255, 0, 0), -1)
                '''平滑处理后的足球位置  效果差'''
                # smoothed_x, smoothed_y = smooth_position('0', soccer_pos[0], soccer_pos[1])
                # cv2.circle(arc, (smoothed_x, smoothed_y), 6, (255, 0, 0), -1)
                # cv2.circle(arc, (smoothed_x, smoothed_y), 10, (204,51,255), -1)
        if i in persons_keys:
            for person_pos in list(persons_pos[i].values())[0]:
                '''将球员坐标画到球二维图中'''
                ID = person_pos[0]
                x = person_pos[1]
                y = person_pos[2]
                x, y = smooth_position(ID, x, y)
                cv2.circle(arc, (x, y), 10, counters[ID].get_most_frequent_color()[0], -1)
                cv2.putText(arc, person_pos[0], (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            counters[ID].get_most_frequent_color()[0], 1)
                if ID in speed_dic:
                    index = speed_cnt[ID]
                    # print(speed_dic[ID][0][index])
                    '''画出对应球员ID的速度'''
                    cv2.putText(arc, str(speed_dic[ID][0][index]), (x + 10, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    speed_cnt[ID] = index + 1

        print(i)
        videoWriter.write(arc)

    cap.release()
    videoWriter.release()

def linear_interpolation(frames):
    interpolated_frames = []
    for i in range(len(frames)):
        if frames[i]:  # 如果当前帧有数据
            interpolated_frames.append(frames[i])
        elif i == 0:
            interpolated_frames.append({})
            continue
        else:
            # 当前帧没有数据，需要进行插值
            prev_frame = None
            next_frame = None
            # 找到前一帧和后一帧
            for j in range(i - 1, -1, -1):
                if frames[j]:
                    prev_frame = frames[j]
                    break
            for k in range(i + 1, len(frames)):
                if frames[k]:
                    next_frame = frames[k]
                    break
            # 如果找到了前一帧和后一帧，进行线性插值
            if prev_frame and next_frame:
                interpolated_frame = {}
                for key_pre in prev_frame.keys():
                    for key_next in next_frame:

                        x_prev, y_prev, w_pre, h_pre = prev_frame[key_pre][0]
                        x_next, y_next, w_next, h_next = next_frame[key_next][0]
                        # 线性插值计算当前帧的位置
                        x_interp = int(x_prev + (x_next - x_prev) * (i - j) / (k - j))
                        y_interp = int(y_prev + (y_next - y_prev) * (i - j) / (k - j))
                        w_interp = int(w_pre + (w_next - w_pre) * (i - j) / (k - j))
                        h_interp = int(h_pre + (h_next - h_pre) * (i - j) / (k - j))
                        interpolated_frame[i] = [(x_interp, y_interp, w_interp, h_interp)]
                interpolated_frames.append(interpolated_frame)
            else:
                interpolated_frames.append({})
    return interpolated_frames

import numpy as np

if __name__ == "__main__" :
    # todo
    start_time = time.time()
    # 二维图大小
    template_h = 800
    template_w = 1280
    # 导出可视化数据路径
    excel_path = 'outputs/data.xlsx'
    # 导入视频路径
    video_path = r"inputs/test.mp4"
    # 导出视频路径
    soccer_persons_video_path = r"outputs/vis.mp4"
    # 导出二维图视频路径
    soccer_persons_2D_path = "outputs/vis_2d.mp4"
    # 二维图模板
    img_dir = 'inputs/field.png'

    cap = cv2.VideoCapture(video_path)
    # 获取视频的帧率、宽度和高度
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 输出soccer+persons的可视化效果
    #todo model_person model_soccer  要两次加载模型
    # 选择球员模型
    model_person = YOLO(r'models/')
    # 选择足球模型
    model_soccer = torch.hub.load('yolov5-master', 'custom', path=r"models/",
                           source='local')
    persons_pos, pos_speed_all = show_persons(video_path, soccer_persons_video_path)
    # 足球39
    print(pos_speed_all)
    # todo 修改足球置信度
    pos_soccers = get_pos(video_path)
    print(pos_soccers)
    # 线性插值足球位置
    pos_soccers=linear_interpolation(pos_soccers)

    soccers_pos = show_soccer(soccer_persons_video_path, pos_soccers)

    # 输出soccer+persons的二维图
    soccer_persons_2D(soccers_pos, persons_pos, img_dir, excel_path, video_path, soccer_persons_2D_path,pos_speed_all)

    print("---total: %s seconds ---" % (time.time() - start_time))
