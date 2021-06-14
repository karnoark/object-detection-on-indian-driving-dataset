# -*- coding: utf-8 -*-
"""indian driving  dataset v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rPaokOB_VKQCOfnaBnK74JoqcmPVI5pX
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  
# %cd yolov5

!nvidia-smi

# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch

from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!curl -L "https://app.roboflow.com/ds/Y4exlFspST?key=FBsyjEmvum" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# Commented out IPython magic to ensure Python compatibility.
# %cat data.yaml

# define number of classes based on YAML
import yaml
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# Commented out IPython magic to ensure Python compatibility.
# %cat /content/yolov5/models/yolov5l.yaml

#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

# Commented out IPython magic to ensure Python compatibility.
# %%writetemplate /content/yolov5/models/custom_yolov5l.yaml
# 
# # parameters
# nc: {num_classes}  # number of classes
# depth_multiple: 1.0  # model depth multiple
# width_multiple: 1.0  # layer channel multiple
# 
# # anchors
# anchors:
#   - [10,13, 16,30, 33,23]  # P3/8
#   - [30,61, 62,45, 59,119]  # P4/16
#   - [116,90, 156,198, 373,326]  # P5/32
# 
# # YOLOv5 backbone
# backbone:
#   # [from, number, module, args]
#   [[-1, 1, Focus, [64, 3]],  # 0-P1/2
#    [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
#    [-1, 3, C3, [128]],
#    [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
#    [-1, 9, C3, [256]],
#    [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
#    [-1, 9, C3, [512]],
#    [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
#    [-1, 1, SPP, [1024, [5, 9, 13]]],
#    [-1, 3, C3, [1024, False]],  # 9
#   ]
# 
# # YOLOv5 head
# head:
#   [[-1, 1, Conv, [512, 1, 1]],
#    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#    [[-1, 6], 1, Concat, [1]],  # cat backbone P4
#    [-1, 3, C3, [512, False]],  # 13
# 
#    [-1, 1, Conv, [256, 1, 1]],
#    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#    [[-1, 4], 1, Concat, [1]],  # cat backbone P3
#    [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
# 
#    [-1, 1, Conv, [256, 3, 2]],
#    [[-1, 14], 1, Concat, [1]],  # cat head P4
#    [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
# 
#    [-1, 1, Conv, [512, 3, 2]],
#    [[-1, 10], 1, Concat, [1]],  # cat head P5
#    [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
# 
#    [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
#   ]

# Commented out IPython magic to ensure Python compatibility.
# %%time
# %cd /content/yolov5/
# !python train.py --img 416 --batch 64 --epochs 40 --data '../data.yaml' --cfg ./models/custom_yolov5l.yaml --weights '' --name yolov5l_results  --cache

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir runs

print("GROUND TRUTH TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/yolov5s_results/test_batch0_labels.jpg', width=900)

print("GROUND TRUTH AUGMENTED TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/yolov5s_results/train_batch0.jpg', width=900)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov5/

!python detect.py --weights runs/train/yolov5l_results/weights/best.pt --img 416 --conf 0.4 --source ../test/images

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cp /content/yolov5/runs/train/yolov5s_results/weights/best.pt /content/gdrive/My\ Drive

!python detect.py --weights runs/train/yolov5l_results2/weights/best.pt --conf 0.49 --source ../newroad.mp4

# Commented out IPython magic to ensure Python compatibility.
# %cp /content/yolov5/runs/detect/exp13/newroad.mp4 /content/gdrive/My\ Drive

