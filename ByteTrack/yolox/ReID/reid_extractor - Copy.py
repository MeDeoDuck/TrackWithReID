# reid_extractor.py
import torch, cv2, numpy as np
from torchvision import transforms
from model import make_model
from yacs.config import CfgNode as CN
try:
    from config import cfg as reid_cfg
except Exception:
    reid_cfg = None

class ReIDExtractor:
    def __init__(self, weight_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if reid_cfg is not None:
            cfg = reid_cfg.clone()
            cfg.defrost()
            cfg.MODEL.NAME = 'transformer'
            cfg.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
            cfg.MODEL.PRETRAIN_CHOICE = 'none'
            cfg.MODEL.PRETRAIN_PATH = ''
            cfg.INPUT.SIZE_TEST = [224, 224]
            cfg.INPUT.SIZE_TRAIN = [224, 224]
            cfg.MODEL.JPM = True
            cfg.MODEL.RE_ARRANGE = True
            cfg.freeze()
        else:
            cfg = CN()
            cfg.MODEL = CN()
            cfg.MODEL.NAME = 'transformer'
            cfg.MODEL.SIE_COE = 3.0
            cfg.MODEL.DROP_PATH = 0.0
            cfg.MODEL.STRIDE_SIZE = [16, 16]
            cfg.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
            cfg.INPUT = CN()
            cfg.INPUT.SIZE_TRAIN = [224, 224]
            cfg.INPUT.SIZE_TEST  = [224, 224]

        self.model = make_model(cfg, num_class=0, camera_num=0, view_num=0)

        state = torch.load(weight_path, map_location='cpu')
        if isinstance(state, dict):
            for k in ['state_dict', 'model', 'net', 'checkpoint']:
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break

        state = {k.replace('module.', ''): v for k, v in state.items()}

        model_state = self.model.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        print("[ReID] load_state_dict -> missing:", len(missing), "unexpected:", len(unexpected))
        self.model.to(self.device).eval()

        # === FP16 (GPU) ===
        self.use_half = (self.device.type == 'cuda')
        if self.use_half:
            self.model.half()

        self.h, self.w = 224, 224
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _preproc(self, img_bgr, x1y1x2y2, pad=10):
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = x1y1x2y2
        x1 = max(0, int(x1-pad)); y1 = max(0, int(y1-pad))
        x2 = min(w-1, int(x2+pad)); y2 = min(h-1, int(y2+pad))
        crop = img_bgr[y1:y2+1, x1:x2+1]
        if crop.size == 0:
            crop = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.tf(crop)

    @torch.no_grad()
    def __call__(self, img_bgr, boxes):
        """
        boxes: (N,4) x1y1x2y2 in original image space
        returns: (N, D) L2-normalized features (torch.float32 on CPU)
        """
        if len(boxes) == 0:
            return torch.empty(0, 768)
        batch = torch.stack([self._preproc(img_bgr, b) for b in boxes], dim=0).to(self.device)

        if self.use_half:
            batch = batch.half()

        feats = self.model(batch)              # (N, D)
        feats = torch.nn.functional.normalize(feats, dim=1)
        feats = feats.float() 
        return feats.detach().cpu()
