# Online-Realtime-Action-Recognition-based-on-OpenPose
A skeleton-based real-time online action recognition project, classifying and recognizing base on framewise joints, which can be used for safety monitoring..
(The code comments are partly descibed in chinese)


------
## Introduction
*The **pipline** of this work is:*
 - Realtime pose estimation by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose);
 - Online human tracking for multi-people scenario by [DeepSort algorithm](https://github.com/nwojke/deep_sortv);
 - Action recognition with DNN for each person based on single framewise joints detected from Openpose.


------
## Dependencies
 - python >= 3.5
 - Opencv >= 3.4.1
 - sklearn
 - tensorflow & keras
 - numpy & scipy
 - pathlib

---

## Required Installation
* NVIDIA graphic card driver
* CUDA toolkit (need to match the version of the driver)
* CUDNN (need to match the version of the CUDA)
* `$ pip3 install tensorflow-gpu`
* `$ git clone https://github.com/zonghan0904/Online-Realtime-Action-Recognition-based-on-OpenPose.git`
* Follow the installation instructions on README

------
## Usage
 - Download the openpose VGG tf-model with command line `./download.sh`(/Pose/graph_models/VGG_origin) or fork [here](https://pan.baidu.com/s/1XT8pHtNP1FQs3BPHgD5f-A#list/path=%2Fsharelink1864347102-902260820936546%2Fopenpose%2Fopenpose%20graph%20model%20coco&parentPath=%2Fsharelink1864347102-902260820936546), and place it under the corresponding folder;
 **VGG_origin**: training with the VGG net, as same as the CMU providing caffemodel, more accurate but slower, **mobilenet_thin**:  training with the Mobilenet, much smaller than the origin VGG, faster but less accurate.
 **However, Please attention that the Action Dataset in this repo is collected along with the** ***VGG model*** **running**.

### For WebCam
 - `python save_video.py`, it will **start the webcam and save the video**.
 - `python collect_data.py`, it will **start the webcam and generate the joints data (training data) per frame as a txt file**.
 (you can choose to test video with command `python collect_data.py --video=test.mp4`)
 - `python test_webcam.py`, it will **start the webcam and classify actions**.
 (you can choose to test video with command `python test_webcam.py --video=test.mp4`)

### For AE450
 - `python test_AE450.py`, it will **start AE450 and classify actions**.


------
## Training with own dataset
 - prepare data(actions) by running `collect_data.py`, the origin data will be saved as a `.txt`.
 - transforming the `.txt` to `.csv`, you can use EXCEL to do this.
 - do the training with the `traing.py` in `Action/training/`, remember to ***change the action_enum and output-layer of model***.


------

## 檢核項目
![](https://i.imgur.com/uhnxKJj.png)

**※因為Human3.6m資料集需要至[Human3.6M](http://vision.imar.ro/human3.6m/description.php)註冊審核才能下載，因為審核至今仍未通過，因此訓練與驗證皆使用自製資料集進行，之前的進度報告已有提過**

* 輸出資料種類包含：
    * 坐著
    * 揮手
    * 跌倒
    * 其他(無分類)

* 驗收靜態 100 frame 姿態辨識
    * 坐著
	[![sit](https://img.youtube.com/vi/RmOM_BYCvSY/0.jpg)](https://youtu.be/RmOM_BYCvSY)
        * frame 72 判斷成 others
        * frame 73~74 判斷成 fall_down
        * 準確性: 97%
    * 揮手
	[![wave](https://img.youtube.com/vi/oDv26JL-sbo/0.jpg)](https://youtu.be/oDv26JL-sbo)
        * frame 21~23 判斷成 others
        * frame 37~38 判斷成 others
        * frame 55 判斷成 others
        * 準確性: 92%
    * 跌倒
	[![fall_down](https://img.youtube.com/vi/c3aG00qfK88/0.jpg)](https://youtu.be/c3aG00qfK88)
        * frame 23 判斷成 sit
        * frame 26 判斷成 sit
        * frame 63 判斷成 wave
        * frame 74 判斷成 wave
        * frame 89 判斷成 sit
        * 準確性: 95%
    * 其他(無分類)
	[![others](https://img.youtube.com/vi/Cd9-Iv348ow/0.jpg)](https://youtu.be/Cd9-Iv348ow)
        * 準確性: 100%
    * **輸出資料頻率可參考影片右上角，平均頻率有高於10Hz**

---

## Documents

### For AE450

#### Subscribe
* ROS Topic of raw image: /camera0/color/image_raw
* ROS Topic of depth image: /camera0/aligned_depth_to_color/image_raw
(the default height and default width of the depth image are not the same as raw image, ==need to be aligned== with `roslaunch realsense2_camera rs_camera.launch align_depth:=true`)

#### Publish
* ROS Topic of results image: /ae450/image/color
* ROS Topic of ROI: /ae450/image/roi (set x_offset=pixel_x, set y_offset=pixel_y, set heigh=depth)

---

## How to Train?

![](https://i.imgur.com/CCC6Vbr.png)


![](https://i.imgur.com/1HbMJEZ.png)


1. 根據需要辨識的動作種類拍攝影片，如下圖所示，假設需要辨識4種分類，那麼至少需要拍攝4種分類的影片，每種動作分類的影片最好符合下列條件：

    * 單一被拍攝者，出現多重被拍攝者會影響到後續訓練資料集的蒐集
    * 背景盡量較單純，避免OpenPose錯誤偵測到joint而影響到後續訓練資料集的蒐集
    * 被拍攝者應盡量平均出現在影像的各個位置或不同角度

    ![](https://i.imgur.com/MW1helU.jpg)

2. 執行 `python collect_data.py --video={動作類別}.mp4` 可以生成被拍攝者在影片中所有執行該類別動作時的joint data，而該joint data會以與影片同名字的txt形式產生

    ![](https://i.imgur.com/cUjkZKi.png)

3. 以`empty_data.csv`作為樣板複製一份`ncrl_data_reduced.csv`，將所有的txt檔改檔名成csv檔並將內容依序複製貼上到`ncrl_data_reduced.csv`上的A欄至AJ欄，並在AK欄填上該動作類別的數字代號，該代號是自己定義。(坐著的影片類別為0，揮手的影片類別為1，跌倒的影片類別為2，其他的影片類別為3)，除此之外，請刪除row和row之間的空白的部份，此部份是因為OpenPose在當下frame未偵測出joint data造成的，可使用圖表的filter功能刪除

    ![](https://i.imgur.com/W8bax2C.png)

4. 若各個csv檔的AL欄後的欄位有值出現，那是因為OpenPose在當下frame若偵測出多個人或是誤判偵測出多個人，則會產生不只一組joint data，請不要複製到`ncrl_data_reduced.csv` (只有`ncrl_data_reduced.csv`的A欄至AK欄會有值，其他欄位不能有值)

    ![](https://i.imgur.com/IikEXNV.png)

5. 將資料集依class的排序由低排到高，另外若OpenPose在當下frame未偵測到對應的joint資料，則會在該欄位填0，為了避免training data有太多沒意義的資料，可以用filter過濾掉出現過多次0的rows，我們所得到的訓練資料過濾掉一行超過12個0的資料，最後存檔後將`ncrl_data_reduced.csv`複製到`Online-Realtime-Action-Recognition-based-on-OpenPose/Action/training`

6. 將`Action/action_enum.py`和`Action/training/train.py`的Actions類別改成要辨識的動作和對應的數字代號，另外可以參考下圖，將`ncrl_data_reduced.csv`裡各個類別的data數量貼上程式裡面的這段

    ![](https://i.imgur.com/qhwOJ5n.png)

7. `python train.py`後就可以開始訓練，所生成的model就會位於`Action/training/`裡面，不需要移動它，`python test_AE450.py`和`python test_webcam.py`會到這個路徑裡面找尋model並進行程式


---

## How to Run?

### AE450

* terminal 1
```
$ roscore
```

* terminal 2 (execute the program)
```
$ python3 test_AE450.py
```

* terminal 3 (play the recorded data)
```
$ rosbag play {the bag file in bag/}

# the name of the raw image topic should be named: /camera0/color/image_raw
# the name of the depth image topic should be named: /camera0/aligned_depth_to_color/image_raw
```

* terminal 4 (results visualization)
```
$ rviz    # subscribe /ae450/image/color and show the image
$ rostopic echo /ae450/image/roi
```

### Video File (only tested on MP4 files)

* terminal 1 (execute the program)
```
$ python3 test_webcam.py --video {path to the video}.mp4

# The results would be displayed on OpenCV figure
```

------
## Acknowledge
Thanks to the following awesome works:
 - [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)
 - [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3)
 - [Real-Time-Action-Recognition](https://github.com/TianzhongSong/Real-Time-Action-Recognition)
