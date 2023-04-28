import cv2
import torch

from advGan import AdvGAN_Attack
from SafeBench.safebench.agent.object_detection.yolov5 import YoloAgent
from SafeBench.safebench.agent.object_detection.utils.general import non_max_suppression 

DEVICE = 'cuda'
EPOCH = 60
LR = 5e-4
TARGET_LABEL = 11
ALPHA = 5
BETA = 1
GAMMA = 1
MAX_PERTURBATION_ALLOWED = 0.01
TARGET_IMAGE = cv2.imread('./stopsign.jpg')
TARGET_IMAGE = torch.from_numpy(TARGET_IMAGE).float().permute(2, 0, 1)
TARGET_IMAGE /= 255.


if len(TARGET_IMAGE.shape) == 3:
    TARGET_IMAGE = TARGET_IMAGE[None]

TARGET_MODEL = YoloAgent(config={'ego_action_dim': 2, 'model_path': "SafeBench/safebench/agent/object_detection/yolov5n.pt", \
                                 'type' : None, 'batch_size' : 1}, logger=None)
TARGET_MODEL.model.eval()
TARGET_MODEL.model.model.eval()
TARGET_MODEL.model.to(DEVICE)

attacker = AdvGAN_Attack(
  device=DEVICE,
  model=TARGET_MODEL.model,
  n_channels=3,
  target_lbl=TARGET_LABEL,
  lr=LR,
  l_inf_bound=MAX_PERTURBATION_ALLOWED,
  alpha=5,
  beta=1,
  gamma=GAMMA,
  n_steps_D=1,
  n_steps_G=1,
  C=MAX_PERTURBATION_ALLOWED ##TODO: NOT SURE THE DIFFERENCE BETWEEN C AND L_INF_BOUND
)


attacker.train(TARGET_IMAGE, EPOCH)


# get adversarial image
adv_res = attacker.attack(TARGET_IMAGE)
adv_res = (adv_res.permute(1,2,0).numpy())*255.
## TODO: multiply 255 back ???
cv2.imwrite("adv_stop_sign.jpg", adv_res)