import os, glob, cv2, torch, numpy as np

class IdentityBank:
    def __init__(self, name2feats: dict[str, torch.Tensor]):
        self.names = list(name2feats.keys())
        protos = []
        for name in self.names:
            f = name2feats[name]                     # (K, D)
            f = torch.as_tensor(f, dtype=torch.float32)
            f = torch.nn.functional.normalize(f, dim=1)
            proto = torch.nn.functional.normalize(f.mean(dim=0), dim=0)
            protos.append(proto)
        self.protos = torch.stack(protos, dim=0)     # (M, D), CPU float32

    def match(self, feat, thresh=0.45):
        if feat is None:
            return None, 0.0
        f = torch.as_tensor(feat, dtype=torch.float32)
        if f.ndim > 1:
            f = f.reshape(-1)
        f = torch.nn.functional.normalize(f, dim=0)
        sims = (self.protos @ f)                     # (M,)
        best = int(torch.argmax(sims))
        best_sim = float(sims[best])
        if best_sim >= thresh:
            return self.names[best], best_sim
        return None, best_sim


def build_identity_bank_from_folders(gallery_root: str, extractor) -> IdentityBank:
    name2feats = {}
    for name in sorted(os.listdir(gallery_root)):
        person_dir = os.path.join(gallery_root, name)
        if not os.path.isdir(person_dir):
            continue
        paths = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            paths.extend(glob.glob(os.path.join(person_dir, ext)))
        feats = []
        for p in sorted(paths):
            img = cv2.imread(p)  # BGR
            if img is None:
                continue
            h, w = img.shape[:2]
            box = np.array([[0, 0, w-1, h-1]], dtype=np.float32)  # Use the whole image
            f = extractor(img, box)        # (1, D), L2-normalized recommended
            if f is not None and len(f) > 0:
                feats.append(torch.as_tensor(f[0], dtype=torch.float32))
        if len(feats) > 0:
            name2feats[name] = torch.stack(feats, dim=0)  # (K, D)
    if len(name2feats) == 0:
        raise RuntimeError("No gallery features found. Check folder structure and images.")
    return IdentityBank(name2feats)
