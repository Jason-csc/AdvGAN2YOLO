import torch
import cv2
import os

DEVICE = 'cuda'
EPOCH = 500
LR_LIST = [5e-5]


# ALPHA_LIST = [10,50,100]
# BETA_LIST = [1,10,50,100]
# GAMMA_LIST = [1000,2000]

ALPHA_LIST = [10]
BETA_LIST = [1]
GAMMA_LIST = [2000]


"""object exsistence attack"""
# MAX_PERTURBATION_ALLOWED = 0.015
# MAX_LPNORM_PERTURBATION_ALLOWED = 0.1
"""targeted (car:3)"""
MAX_PERTURBATION_ALLOWED = 0.05
MAX_LPNORM_PERTURBATION_ALLOWED = 15
"""untargeted attack"""
# MAX_PERTURBATION_ALLOWED = 0.01
# MAX_LPNORM_PERTURBATION_ALLOWED = 0.1


TARGET_IMAGE = cv2.imread('./stopsign.jpg')
TARGET_IMAGE = torch.from_numpy(TARGET_IMAGE).float().permute(2, 0, 1)
TARGET_IMAGE /= 255.

if len(TARGET_IMAGE.shape) == 3:
    TARGET_IMAGE = TARGET_IMAGE[None]

TARGET_IMAGE_LABEL = 11


ADVIMG_PATH = './results/advimages/'
if not os.path.exists(ADVIMG_PATH):
    os.makedirs(ADVIMG_PATH)

MODELS_PATH = './checkpoints/AdvGAN/'
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

LOSSES_PATH = './results/losses/'
if not os.path.exists(LOSSES_PATH):
    os.makedirs(LOSSES_PATH)


# class names
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']


ATTACK_MODE = 2
"""
-1  : untargeted attack
-2  : attack confidence score for object existence
>=0 : targeted attack indicating the target label  (3 for "car" label)
"""

ATTACK_INFO = None
if ATTACK_MODE == -1:
    ATTACK_INFO = "Untargeted"
elif ATTACK_MODE == -2:
    ATTACK_INFO = "NoObject"
elif 0 <= ATTACK_MODE < len(LABELS):
    ATTACK_INFO = f"Targeted_{LABELS[ATTACK_MODE]}"
else:
    assert False, print("Wrong Attack mode")

print(f"Attack Type: {ATTACK_INFO}")
