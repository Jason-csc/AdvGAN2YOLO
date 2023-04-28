import cv2
import torch

from model import AdvGAN_Attack
from SafeBench.safebench.agent.object_detection.yolov5 import YoloAgent

TARGET_MODEL = YoloAgent(config={'ego_action_dim': 2, 'model_path': None, \
                                 'type' : None, 'batch_size' : 1}, logger=None)
TARGET_MODEL.model.eval()


EPOCH = 60
LR = 5e-4
TARGET_LABEL = 11
ALPHA = 5
BETA = 1
GAMMA = 1
MAX_PERTURBATION_ALLOWED = 0.01
TARGET_IMAGE = torch.from_numpy(cv2.imread('./stopsign.jpg'))
TARGET_IMAGE = torch.from_numpy(TARGET_IMAGE).float().permute(2, 0, 1)
TARGET_IMAGE /= 255.
if len(TARGET_IMAGE.shape) == 3:
    TARGET_IMAGE = TARGET_IMAGE[None]


attacker = AdvGAN_Attack(
  device='cuda',
  model=TARGET_MODEL.model,
  n_channels=3,
  target_lbl=TARGET_LABEL,
  lr=LR,
  l_inf_bound=MAX_PERTURBATION_ALLOWED,
  alpha=5,
  beta=1,
  gamma=GAMMA,
  n_steps_D=1,
  n_steps_G=1
)


attacker.train(TARGET_IMAGE, EPOCH)