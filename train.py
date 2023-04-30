import cv2
import torch

from advGan import AdvGAN_Attack
from SafeBench.safebench.agent.object_detection.yolov5 import YoloAgent

DEVICE = 'cuda'
EPOCH = 60
LR = 5e-4
TARGET_LABEL = 11
ALPHA = 100
BETA = 1
GAMMA = 1
MAX_PERTURBATION_ALLOWED = 0.01
TARGET_IMAGE = cv2.imread('./stopsign.jpg')
TARGET_IMAGE = torch.from_numpy(TARGET_IMAGE).float().permute(2, 0, 1)
TARGET_IMAGE /= 255.
MAX_LPNORM_PERTURBATION_ALLOWED = 0.1

torch.autograd.set_detect_anomaly(True)

if len(TARGET_IMAGE.shape) == 3:
    TARGET_IMAGE = TARGET_IMAGE[None]

TARGET_MODEL = YoloAgent(config={'ego_action_dim': 2, 'model_path': "SafeBench/safebench/agent/object_detection/yolov5n.pt", \
                                 'type' : None, 'batch_size' : 1}, logger=None)
TARGET_MODEL.model.eval()
TARGET_MODEL.model.model.eval()
TARGET_MODEL.model.to(DEVICE)

attacker = AdvGAN_Attack(
  device=DEVICE,
  target_model=TARGET_MODEL,
  n_channels=3,
  target_lbl=TARGET_LABEL,
  lr=LR,
  l_inf_bound=MAX_PERTURBATION_ALLOWED,
  alpha=ALPHA,
  beta=BETA,
  gamma=GAMMA,
  n_steps_D=1,
  n_steps_G=1,
  C=MAX_LPNORM_PERTURBATION_ALLOWED, ##TODO: NOT SURE THE DIFFERENCE BETWEEN C AND L_INF_BOUND
  is_relativistic=True
)

attacker.train(TARGET_IMAGE, EPOCH)


# get adversarial image
res = attacker.evaluate(TARGET_IMAGE)
adv_res = res["adv_image"][0]
adv_res = (adv_res.permute(1,2,0).cpu().numpy())*255
cv2.imwrite("adv_stop_sign.jpg", adv_res)

print("Evaluation Info:")
print(res["evaluation"])