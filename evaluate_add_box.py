import numpy as np
import cv2
import torch
torch.manual_seed(0)

from itertools import product

from advGan import AdvGAN_Attack
from SafeBench.safebench.agent.object_detection.yolov5 import YoloAgent

DEVICE = 'cuda'

def evaluate(image_path):
	IMAGE = cv2.imread(image_path)
	IMAGE = torch.from_numpy(IMAGE).float().permute(2, 0, 1)
	IMAGE /= 255.
	IMAGE = IMAGE[None]
	res = attacker.evaluate(IMAGE, preturb=False)
	labels = res["evaluation"][1][0]['labels'][:5]
	scores = res["evaluation"][1][0]['scores'][:5]
	boxes = res["evaluation"][1][0]['boxes'][:5]
	obj_existence = res["evaluation"][0][:, :, 4].max(1)[0].item()
	return IMAGE, labels, scores, boxes, obj_existence


def add_box(image_path, labels, scores, boxes):
	IMAGE = cv2.imread(image_path)
	clist = [(240,0,0),(0,180,0)]
	for (x1,y1,x2,y2), s, lbl, color in zip(boxes[:2], scores[:2], labels[:2], clist[:2]):
		x1 = int(x1//2)
		y1 = int(y1//2)
		x2 = int(x2//2)
		y2 = int(y2//2)
		cv2.rectangle(IMAGE, (x1, y1), (x2, y2), color, 2)
		text = f"{lbl}:{round(s.item(),2)}"
		cv2.putText(IMAGE, text, (x1, max(y1-10,24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
	cv2.imwrite("add_box_"+image_path.split('/')[-1], IMAGE)



TARGET_MODEL = YoloAgent(config={'ego_action_dim': 2, 
                                 'model_path': "yolov5x6.pt", \
                                #  'model_path': "SafeBench/safebench/agent/object_detection/yolov5n.pt", \
                                 'type' : None, 'batch_size' : 1}, logger=None)
TARGET_MODEL.model.eval()
TARGET_MODEL.model.model.eval()
TARGET_MODEL.model.to(DEVICE)


attacker = AdvGAN_Attack(
  	  target_model=TARGET_MODEL,
	  device=DEVICE
  	)



# TARGET_IMAGE, labels, scores, boxes, obj_existence = evaluate('stopsign.jpg')
# print("Check labels",labels)
# print("Check scores",scores)
# print("Check boxes",boxes)
# print("Check obj_existence",obj_existence)

# print('add Box')
# add_box("stopsign.jpg", labels, scores, boxes)

# TARGET_IMAGE2 = np.load('results/advimages/NoObject_adv_stop_sign_10_1_1500_0.0001_stopsign_152.8.npy')
# TARGET_IMAGE2 = cv2.imread('results/advimages/Targeted_car_adv_stop_sign_10_50_800_5e-05_car_12.92.png')
adv_image_path = 'results/advimages/Untargeted_adv_stop_sign_10_1_1000_5e-05_scissors_8.75.png'
# adv_image_path = 'results/advimages/Targeted_car_adv_stop_sign_10_50_800_5e-05_car_12.92.png'

TARGET_IMAGE2, labels, scores, boxes, obj_existence = evaluate(adv_image_path)
print("Check labels",labels)
print("Check scores",scores)
print("Check boxes",boxes)
print("Check obj_existence",obj_existence)

print('add Box')
add_box(adv_image_path, labels, scores, boxes)

# print("Perturb Norm")
# perturbation = TARGET_IMAGE2 - TARGET_IMAGE
# print(perturbation.shape)
# norm = torch.mean(torch.norm(perturbation.reshape(perturbation.shape[0], -1), 2, dim=1))
# print(norm)



