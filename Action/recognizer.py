# -*- coding: UTF-8 -*-
import numpy as np
import cv2 as cv
from pathlib import Path
from Tracking.deep_sort import preprocessing
from Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Tracking.deep_sort.detection import Detection
from Tracking import generate_dets as gdet
from Tracking.deep_sort.tracker import Tracker
from keras.models import load_model
from .action_enum import Actions

from cv_bridge import CvBridge
bridge = CvBridge()

# Use Deep-sort(Simple Online and Realtime Tracking)
# To track multi-person for multi-person actions recognition

# 定义基本参数
file_path = Path.cwd()
clip_length = 15
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
PRED_THRESHOLD = 0.51

# 初始化deep_sort
model_filename = str(file_path/'Tracking/graph_model/mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# track_box颜色
trk_clr = (0, 255, 0)


# class ActionRecognizer(object):
#     @staticmethod
#     def load_action_premodel(model):
#         return load_model(model)
#
#     @staticmethod
#     def framewise_recognize(pose, pretrained_model):
#         frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
#         joints_norm_per_frame = np.array(pose[-1])
#
#         if bboxes:
#             bboxes = np.array(bboxes)
#             features = encoder(frame, bboxes)
#
#             # score to 1.0 here).
#             detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]
#
#             # 进行非极大抑制
#             boxes = np.array([d.tlwh for d in detections])
#             scores = np.array([d.confidence for d in detections])
#             indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
#             detections = [detections[i] for i in indices]
#
#             # 调用tracker并实时更新
#             tracker.predict()
#             tracker.update(detections)
#
#             # 记录track的结果，包括bounding boxes及其ID
#             trk_result = []
#             for trk in tracker.tracks:
#                 if not trk.is_confirmed() or trk.time_since_update > 1:
#                     continue
#                 bbox = trk.to_tlwh()
#                 trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
#                 # 标注track_ID
#                 trk_id = 'ID-' + str(trk.track_id)
#                 cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)
#
#             for d in trk_result:
#                 xmin = int(d[0])
#                 ymin = int(d[1])
#                 xmax = int(d[2]) + xmin
#                 ymax = int(d[3]) + ymin
#                 # id = int(d[4])
#                 try:
#                     # xcenter是一帧图像中所有human的1号关节点（neck）的x坐标值
#                     # 通过计算track_box与human的xcenter之间的距离，进行ID的匹配
#                     tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
#                     j = np.argmin(tmp)
#                 except:
#                     # 若当前帧无human，默认j=0（无效）
#                     j = 0
#
#                 # 进行动作分类
#                 if joints_norm_per_frame.size > 0:
#                     joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
#                     joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
#                     pred = np.argmax(pretrained_model.predict(joints_norm_single_person))
#                     init_label = Actions(pred).name
#                     # 显示动作类别
#                     cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)
#                 # 画track_box
#                 cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
#         return frame

def load_action_premodel(model):
    return load_model(model)

def framewise_recognize(pose, pretrained_model, depth_image=None):
    min_depth = np.inf
    nearest_person = None
    frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
    joints_norm_per_frame = np.array(pose[-1])

    if bboxes:
        bboxes = np.array(bboxes)
        features = encoder(frame, bboxes)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

        # 进行非极大抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # 调用tracker并实时更新
        tracker.predict()
        tracker.update(detections)

        # 记录track的结果，包括bounding boxes及其ID
        trk_result = []
        for trk in tracker.tracks:
            if not trk.is_confirmed() or trk.time_since_update > 1:
                continue
            bbox = trk.to_tlwh()
            trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
            # 标注track_ID
            trk_id = 'ID-' + str(trk.track_id)
            cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)

        for d in trk_result:
            xmin = int(d[0])
            ymin = int(d[1])
            xmax = int(d[2]) + xmin
            ymax = int(d[3]) + ymin
            # id = int(d[4])
            try:
                # xcenter是一帧图像中所有human的1号关节点（neck）的x坐标值
                # 通过计算track_box与human的xcenter之间的距离，进行ID的匹配
                tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                j = np.argmin(tmp)
            except:
                # 若当前帧无human，默认j=0（无效）
                j = 0

            # 进行动作分类
            if joints_norm_per_frame.size > 0:
                joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
                joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
                if np.count_nonzero(joints_norm_single_person) > 20:
                    probs = pretrained_model.predict(joints_norm_single_person)
                    # print(probs)
                    max_score = np.max(probs)
                    if max_score < PRED_THRESHOLD:
                        init_label = "others"
                    else:
                        pred = np.argmax(probs)
                        init_label = Actions(pred).name
                else:
                    init_label = "others"
                    cv.putText(frame, 'WARNING: not enough joints to recognize action!',
                               (20, 60), cv.FONT_HERSHEY_SIMPLEX, 1.5, (30, 105, 210), 4)

                # 显示动作类别
                cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)
                # 异常预警(under scene)
                if init_label == 'fall_down':
                    cv.putText(frame, 'WARNING: someone is falling down!', (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 255), 4)
            # 画track_box
            cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)

            if depth_image is not None:
                detected_joints_count = 0
                detected_joints_depth_sum = 0
                detected_joints_x_sum = 0
                detected_joints_y_sum = 0
                for joints_id in range(18):
                    try:
                        detected_joints_x_sum += joints[0][joints_id][0]
                        detected_joints_y_sum += joints[0][joints_id][1]
                        detected_joints_depth_sum += depth_image[joints[0][joints_id][1], joints[0][joints_id][0]]
                        detected_joints_count += 1
                    except:
                        # joints_id is not detected
                        pass
                if detected_joints_count != 0:
                    joints_average_x = detected_joints_x_sum // detected_joints_count
                    joints_average_y = detected_joints_y_sum // detected_joints_count
                    joints_average_depth = detected_joints_depth_sum // detected_joints_count

                    depth_image_height, depth_image_width = depth_image.shape
                    joints_average_x = np.clip(joints_average_x, 0, depth_image_width-1)
                    joints_average_y = np.clip(joints_average_y, 0, depth_image_height-1)
                    joints_average_center = (joints_average_x, joints_average_y)
                    # cv.circle(frame, joints_average_center, 3, (0, 0, 255), 10)   # center of human
                else:
                    joints_average_center = (0, 0)
                    joints_average_depth = 0

                if joints_average_depth < min_depth:
                    min_depth = joints_average_depth
                    nearest_person = (joints_average_center[0], joints_average_center[1], min_depth)

    return frame, nearest_person

