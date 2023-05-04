from yolo_safebench import YoloModel
from advGan import AdvGAN_Attack
from itertools import product
import os
import numpy as np
import random
import cv2
import torch
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from config import EPOCH, LR_LIST, DEVICE, ALPHA_LIST, BETA_LIST, GAMMA_LIST, \
                   TARGET_IMAGE, TARGET_IMAGE_LABEL,\
                   MAX_PERTURBATION_ALLOWED, MAX_LPNORM_PERTURBATION_ALLOWED,\
                   ATTACK_MODE, ATTACK_INFO,\
                   ADVIMG_PATH



TARGET_MODEL = YoloModel(config={'ego_action_dim': 2,
                                 'model_path': "yolov5n6.pt", \
                                 #  'model_path': "SafeBench/safebench/agent/object_detection/yolov5n.pt", \
                                 'type': None, 'batch_size': 1}, logger=None)
TARGET_MODEL.model.eval()
TARGET_MODEL.model.model.eval()
TARGET_MODEL.model.to(DEVICE)


for i, (LR, ALPHA, BETA, GAMMA) in enumerate(product(LR_LIST, ALPHA_LIST, BETA_LIST, GAMMA_LIST)):
    print(f"LR {LR} ALPHA {ALPHA} BETA {BETA} GAMMA {GAMMA}")
    attacker = AdvGAN_Attack(
        device=DEVICE,
        target_model=TARGET_MODEL,
        n_channels=3,
        target_img_lbl=TARGET_IMAGE_LABEL,
        lr=LR,
        l_inf_bound=MAX_PERTURBATION_ALLOWED,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        n_steps_D=1,
        n_steps_G=1,
        C=MAX_LPNORM_PERTURBATION_ALLOWED,
        attack_mode=ATTACK_MODE,
        is_relativistic=True
    )

    if i == 0:
        print("Predicting original image")
        res = attacker.evaluate(TARGET_IMAGE, preturb=False)
        labels = res["evaluation"][1][0]['labels'][:5]
        scores = res["evaluation"][1][0]['scores'][:5]
        obj_existence = res["evaluation"][0][:, :, 4].max(1)[0].item()
        print("Check labels",labels)
        print("Check scores",scores)
        print("Check obj_existence",obj_existence)

    best_norm, best_epoch, best_labels, best_image, best_score, best_obj_existence = attacker.train(TARGET_IMAGE, EPOCH)
    
    print("="*10)
    print(f"LR{LR} ALPHA {ALPHA} BETA {BETA} GAMMA {GAMMA}")
    print("best norm", best_norm)
    print("best epoch", best_epoch)
    print("best labels", best_labels)
    print("best_score", best_score)
    print("best_obj_existence", best_obj_existence)
    print("="*10)

    # store adversarial image
    if not best_image is None:
        adv_res = best_image[0]
        adv_res = (adv_res.permute(1, 2, 0).cpu().numpy())*255
        np.save(ADVIMG_PATH + f"{ATTACK_INFO}_adv_stop_sign_{ALPHA}_{BETA}_{GAMMA}_{LR}_{best_labels[0]}_{round(best_norm.item(),2)}.npy", adv_res)
        cv2.imwrite(ADVIMG_PATH +f"{ATTACK_INFO}_adv_stop_sign_{ALPHA}_{BETA}_{GAMMA}_{LR}_{best_labels[0]}_{round(best_norm.item(),2)}.png", adv_res)
