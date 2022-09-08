# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
from utils import load_pretrain_model
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
from sensor_msgs.msg import RegionOfInterest

import rospy
from sensor_msgs.msg import Image
import numpy as np
import sys
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow.compat.v2 as tf

from cv_bridge import CvBridge
bridge = CvBridge()

def set_video_writer(write_fps=15):
    out_file_path = "./test_out/AE450.mp4"
    print(out_file_path)
    return cv.VideoWriter(out_file_path,
                          cv.VideoWriter_fourcc(*'mp4v'),
                          write_fps,
                          (1280, 720))

class CameraReader:
    def __init__(self):
        # self.sub_image = rospy.Subscriber("/device_0/sensor_1/Color_0/image/data", Image, self.image_cb)
        self.sub_image = rospy.Subscriber("/camera0/color/image_raw", Image, self.image_cb)
        self.sub_depth_image = rospy.Subscriber("/camera0/aligned_depth_to_color/image_raw", Image, self.depth_image_cb)
        self.image = None
        self.depth_image = None
        self.rate = rospy.Rate(30)
        self._initialize()

    def image_cb(self, msg):
        self.header = msg.header
        self.image = bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_image_cb(self, msg):
        self.depth_image = bridge.imgmsg_to_cv2(msg)

    def _initialize(self):
        while self.image is None:
            self.rate.sleep()

class LiteModel:

    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]

rospy.init_node("test_ae450_node")
image_publisher = rospy.Publisher("/ae450/image/color", Image, queue_size=10)
ROI_publisher = rospy.Publisher("/ae450/image/roi", RegionOfInterest, queue_size=10)
camera_reader = CameraReader()

parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# 导入相关模型
estimator = load_pretrain_model('VGG_origin')
action_classifier = load_action_premodel('Action/training/ncrl_framewise_recognition.h5')
action_classifier = LiteModel.from_keras_model(action_classifier)

# 参数初始化
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

# # 保存关节数据的txt文件，用于训练过程(for training)
# f = open('origin_data.txt', 'a+')
# video_writer = set_video_writer(write_fps=int(7.0))
fps_list = []

last_header = None
while camera_reader.header != last_header:
    last_header, show = camera_reader.header, camera_reader.image
    fps_count += 1
    frame_count += 1

    init_time = time.time()

    # pose estimation
    humans = estimator.inference(show)
    # get pose info
    pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
    # recognize the action framewise
    show, nearest_person = framewise_recognize(pose, action_classifier, camera_reader.depth_image)

    height, width = show.shape[:2]
    # 显示实时FPS值
    if (time.time() - start_time) > fps_interval:
        # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
        realtime_fps = fps_count / (time.time() - start_time)
        fps_list.append(realtime_fps)
        fps_count = 0  # 帧数清零
        start_time = time.time()
    fps_label = 'FPS:{0:.2f}'.format(float(realtime_fps))
    cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    elapsed_time = 'Elapsed Time:{0:.3f} s'.format(float(time.time() - init_time))
    cv.putText(show, elapsed_time, (width-345, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # 显示检测到的人数
    num_label = "Human: {0}".format(len(humans))
    cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # 显示目前的运行时长及总帧数
    if frame_count == 1:
        run_timer = time.time()
    run_time = time.time() - run_timer
    time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
    cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    image_publisher.publish(bridge.cv2_to_imgmsg(show))
    try:
        if nearest_person is not None:
            roi = RegionOfInterest()
            roi.x_offset = nearest_person[0]
            roi.y_offset = nearest_person[1]
            roi.height = nearest_person[2]
            ROI_publisher.publish(roi)
    except Exception as e:
        print("\033[93m" + str(e) + "\033[0m")
    # cv.imshow('Action Recognition based on OpenPose', show)
    # video_writer.write(show)

    # # 采集数据，用于训练过程(for training)
    # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
    # f.write(' '.join(joints_norm_per_frame))
    # f.write('\n')

fps_list = np.array(fps_list)[1:]
print(f"Highest FPS: {fps_list.max()}")
print(f"Lowest FPS: {fps_list.min()}")
print(f"Average FPS: {fps_list.mean()}")
# video_writer.release()
# f.close()
