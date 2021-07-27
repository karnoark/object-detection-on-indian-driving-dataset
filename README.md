# object-detection-on-indian-driving-dataset
## trained yolov5 model on indian driving dataset for 2d object detection

In this repository, I present an application of the latest version of YOLO i.e. YOLOv5, to perform object detection on navigation dataset . I have used the code of [Ultralytics](https://github.com/ultralytics/yolov5) to train the model. Make sure to check their repository also. It's great.

# Introduction

Object detection is a computer vision task that requires object(s) to be detected, localized and classified. In this task, first we need our machine learning model to tell if any object of interest is present in the image. If present, then draw a bounding box around the object(s) present in the image. In the end, the model must classify the object represented by the bounding box. This task requires fast object detection so that it can be implemented in real-time. One of its major applications is its use in real-time object detection in self-driving vehicles.

Joseph Redmon, et al. originally designed YOLOv1, v2 and v3 models that perform real-time object detection. YOLO "You Only Look Once" is a state-of-the-art real-time deep learning algorithm used for object detection, localization and classification in images and videos. This algorithm is very fast, accurate and at the forefront of object detection based projects. 

Each of the versions of YOLO kept improving the previous in accuracy and performance. Then came YOLOv4 developed by another team, further adding to performance of model and finally the YOLOv5 model was introduced by Glenn Jocher in June 2020. This model significantly reduces the model size (YOLOv4 on Darknet had 244MB size whereas YOLOv5 smallest model is of 27MB). YOLOv5 also claims a faster accuracy and more frames per second than YOLOv4 as shown in graph below, taken from Roboflow.ai's website.
![yolov5 comparison](https://github.com/karnoark/object-detection-on-indian-driving-dataset/blob/334e89a37a536001546f36186273562436c5f632/Inferences/yolov5%20comparison.png)


# Dataset

While several datasets for autonomous navigation have become available in recent years, they have tended to focus on structured driving environments. This usually corresponds to well-delineated infrastructure such as lanes, a small number of well-defined categories for traffic participants, low variation in object or background appearance and strong adherence to traffic rules. [Indian Driving Dataset](https://idd.insaan.iiit.ac.in/) novel dataset for road scene understanding in unstructured environments where the above assumptions are largely not satisfied. It consists of 10,000 images, finely annotated with 34 classes collected from 182 drive sequences on Indian roads. The label set is expanded in comparison to popular benchmarks such as Cityscapes, to account for new classes.The dataset consists of images obtained from a front facing camera attached to a car. The car was driven around Hyderabad, Bangalore cities and their outskirts. The images are mostly of 1080p resolution, but there is also some images with 720p and other resolutions.

# Data Generation
I used [roboflow](https://roboflow.com/) for data generation . Roboflow also gives option to generate a dataset based on user defined split. I used 70–20–10 training-validation-test set split. After the data is generated on Roboflow, we get the original images as well as all bounding box locations for all annotated objects in a separate text file for each image, which is convenient. Finally, we get a link to download the generated data with label files. This link contains a key that is restricted to only your account and is not supposed to be shared.

# Process
The code is present in jupyter notebook in attached files. However, it is recommended to copy the whole code in Google Colab notebook.

# Objervations
![Observation using Tensorboard](https://github.com/karnoark/object-detection-on-indian-driving-dataset/blob/334e89a37a536001546f36186273562436c5f632/Inferences/Objervation.png)

# Results
![Inference on Test images](https://github.com/karnoark/object-detection-on-indian-driving-dataset/blob/334e89a37a536001546f36186273562436c5f632/Inferences/yoloV5_idd.jpg

https://user-images.githubusercontent.com/79748668/127192248-27392538-15cb-4957-a2ab-e27d81a2d207.mp4

)
