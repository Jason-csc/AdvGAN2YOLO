from SafeBench.safebench.agent.object_detection.yolov5 import YoloAgent
from SafeBench.safebench.agent.object_detection.utils.general import non_max_suppression
import torch

class YoloModel(YoloAgent):
    def __init__(self, config, logger, train_mode='none') -> None:
          super().__init__(config, logger, train_mode)

    def get_inference(self, image, conf_thres, iou_thres, max_det):
        self.model.eval()
        self.model.model.eval()
        raw_pred = self.model(image, augment=False, visualize=False).detach().cpu()

        pred = non_max_suppression(raw_pred, conf_thres, iou_thres, None, False, max_det=max_det)
        img_annot_list = []
        pred_list = []
        if pred[0].shape[0] == 0:
            pass
        else:
            img_annot_list.append(self.annotate(pred.copy(), image))
        
        pred = self._transform_predictions(pred)
        pred_list.append(pred)
        # TODO: CUDA Memory Management
        torch.cuda.empty_cache()
        return raw_pred, pred_list, img_annot_list