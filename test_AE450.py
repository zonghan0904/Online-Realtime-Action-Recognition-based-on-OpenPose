# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
from utils import load_pretrain_model
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

import rospy
from sensor_msgs.msg import Image
import numpy as np
import sys
import cv2

def imgmsg_to_cv2(img_msg):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    if img_msg.encoding == "rgb8":
        image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

def set_video_writer(write_fps=15):
    out_file_path = "./test_out/AE450.mp4"
    print(out_file_path)
    return cv.VideoWriter(out_file_path,
                          cv.VideoWriter_fourcc(*'mp4v'),
                          write_fps,
                          (1280, 720))

class V:
    def __init__(self):
        # self.sub = rospy.Subscriber("/camera1/color/image_raw", Image, self.image_cb)
        self.sub = rospy.Subscriber("/device_0/sensor_1/Color_0/image/data", Image, self.image_cb)
        self.image = np.zeros((1, 1))

    def image_cb(self, msg):
        self.image = imgmsg_to_cv2(msg)

rospy.init_node("test_node")
v = V()

parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# 导入相关模型
estimator = load_pretrain_model('VGG_origin')
action_classifier = load_action_premodel('Action/training/ncrl_framewise_recognition.h5')

# 参数初始化
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

# # 保存关节数据的txt文件，用于训练过程(for training)
# f = open('origin_data.txt', 'a+')
video_writer = set_video_writer(write_fps=int(7.0))

while cv.waitKey(1) < 0:
    has_frame, show = True, v.image
    if has_frame:
        fps_count += 1
        frame_count += 1

        # pose estimation
        humans = estimator.inference(show)
        # get pose info
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
        # recognize the action framewise
        show = framewise_recognize(pose, action_classifier)

        height, width = show.shape[:2]
        # 显示实时FPS值
        if (time.time() - start_time) > fps_interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 帧数清零
            start_time = time.time()
        fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 显示检测到的人数
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 显示目前的运行时长及总帧数
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv.imshow('Action Recognition based on OpenPose', show)
        video_writer.write(show)

        # # 采集数据，用于训练过程(for training)
        # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        # f.write(' '.join(joints_norm_per_frame))
        # f.write('\n')

video_writer.release()
cap.release()
# f.close()
