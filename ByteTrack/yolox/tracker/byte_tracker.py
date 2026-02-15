import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

# ===== [New] =====
def _cosine_dist_matrix(a, b):
    if not torch.is_tensor(a):
        a = torch.as_tensor(a)
    if not torch.is_tensor(b):
        b = torch.as_tensor(b)
    a = a.float()
    b = b.float()

    b = b.to(a.device)

    if a.numel() == 0 and b.numel() == 0:
        return torch.empty(0, 0, dtype=torch.float32, device=a.device)
    if a.numel() == 0:
        n = b.shape[0] if b.ndim >= 2 else (1 if b.ndim == 1 and b.numel() > 0 else 0)
        return torch.empty(0, n, dtype=torch.float32, device=a.device)
    if b.numel() == 0:
        m = a.shape[0] if a.ndim >= 2 else (1 if a.ndim == 1 and a.numel() > 0 else 0)
        return torch.empty(m, 0, dtype=torch.float32, device=a.device)

    if a.ndim == 1:
        a = a.unsqueeze(0)
    elif a.ndim > 2:
        a = a.view(a.size(0), -1)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    elif b.ndim > 2:
        b = b.view(b.size(0), -1)

    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    sims = a @ b.t()  # (M, N)
    return 1.0 - sims


def _fuse_iou_reid(iou_cost, reid_cost, alpha=0.5, gate=0.2):
    if torch.is_tensor(iou_cost):
        iou = iou_cost.detach().cpu().numpy().astype(np.float32)
    else:
        iou = np.asarray(iou_cost, dtype=np.float32)

    if torch.is_tensor(reid_cost):
        r = reid_cost.detach().cpu().numpy().astype(np.float32)
    else:
        r = np.asarray(reid_cost, dtype=np.float32)

    if iou.size == 0 or r.size == 0:
        return iou

    cos = 1.0 - r

    mask = (cos >= gate).astype(np.float32)
    fused_when_use = alpha * iou + (1.0 - alpha) * r
    fused = mask * fused_when_use + (1.0 - mask) * iou

    return fused

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # ===== [New] ReID fields =====
        self.feat_queue = deque(maxlen=30) #recent embedding buffer
        self.feat_avg = None # Representative embeddings maintained by EMA
        self.name = None # name of identity


    # ===== [New] ReID utils =====
    def update_feat(self, feat, alpha: float=0.5):
        """
        feat: torch.Tensor or np.ndarray, shape (D,) or (1,D) or (D,1)
        - Update self.feat_avg to EMA after L2 normalization internally
        - alpha: EMA coefficient (0~1) => The larger the value, the more recent the value is reflected(Recommended 0.3~0.7)
        """
        if feat is None:
            return
        # to numpy 1D
        if 'torch' in str(type(feat)):
            try:
                feat = feat.detach().cpu().numpy()
            except Exception:
                feat = np.asarray(feat)
        f = np.asarray(feat, dtype=np.float32).reshape(-1)
        # L2 normalizae
        n = np.linalg.norm(f) + 1e-12
        f = f/n

        self.feat_queue.append(f)
        if self.feat_avg is None:
            self.feat_avg = f.copy()
        else:
            self.feat_avg = (1.0 - alpha) * self.feat_avg + alpha * f
            # re-normalize
            nn = np.linalg.norm(self.feat_avg) + 1e-12
            self.feat_avg = self.feat_avg / nn

    def get_feat(self):
        """Return representative embeddings for matching (None possible)"""
        if self.feat_avg is not None:
            return self.feat_avg
        if len(self.feat_queue) > 0:
            return self.feat_queue[-1]
        return None

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, feat=None):
        """Start a new tracklet"""
        if feat is not None:
            self.reid_feat = feat
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        # ===== [NEW] ReID update on init =====
        if feat is not None:
            self.update_feat(feat)

    def re_activate(self, new_track, frame_id, new_id=False, feat=None):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        if feat is not None:
            self.reid_feat = feat
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        # ===== [NEW] ReID update on re-activate =====
        if feat is not None:
            self.update_feat(feat)

    def update(self, new_track, frame_id, feat=None):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        if feat is not None:
            self.reid_feat = feat
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # ===== [NEW] ReID update on match =====
        if feat is not None:
            self.update_feat(feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.reid = None
        self.reid_iou_fuse_alpha = getattr(args, 'reid_iou_alpha', 0.5)
        self.reid_iou_fuse_alpha_second = getattr(args, 'reid_iou_alpha_second', 0.3)
        self.reid_gate = getattr(args, 'reid_gate', 0.3)

    def _assign_identity(self, track, feat):
        """feat(1,D or D)과 id_bank를 비교해 track.name을 채운다."""
        if getattr(self, "id_bank", None) is None or feat is None:
            return

        bank_feats, bank_names = None, None
        ib = self.id_bank
        if isinstance(ib, (list, tuple)) and len(ib) == 2:
            bank_feats, bank_names = ib
        elif isinstance(ib, dict):
            for kf in ("features", "feats", "embeddings"):
                if kf in ib:
                    bank_feats = ib[kf]
                    break
            for kn in ("names", "labels", "ids"):
                if kn in ib:
                    bank_names = ib[kn]
                    break
        if bank_feats is None or bank_names is None:
            return

        bf = torch.as_tensor(bank_feats).float()
        if bf.ndim == 1:
            bf = bf.unsqueeze(0)
        feat = torch.as_tensor(feat).float()
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)

        bf = F.normalize(bf, dim=1)
        feat = F.normalize(feat, dim=1)

        cos = feat @ bf.t()
        score, idx = torch.max(cos.squeeze(0), dim=0)

        gate = getattr(self, "reid_gate", 0.3)
        if float(score.item()) >= float(gate):
            name = bank_names[int(idx.item())]
            try:
                name = str(name)
            except Exception:
                pass
            track.name = name
            track.name_score = float(score.item())
        else:
            if not hasattr(track, "name"):
                track.name = None


    def update(self, output_results, img_info, img_size, ori_img=None):
        self.frame_id += 1

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        keep_idx = np.where(remain_inds)[0]
        second_idx = np.where(inds_second)[0]

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # --- ReID: batch extract once for all detections of this frame ---
        det_feats_all = None
        if (getattr(self, "reid", None) is not None) and (ori_img is not None) and (len(bboxes) > 0):
            det_feats_all = self.reid(ori_img, bboxes.astype(np.float32))  # (N,D) torch tensor (CPU)

        # --- wrap detections as STrack (1st set) ---
        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # --- split confirmed/unconfirmed ---
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # =========================
        # Step 2: First association
        # =========================
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)

        # ---- IoU + ReID fusion (1st) ----
        if (det_feats_all is not None) and (len(detections) > 0):
            det_feats_keep = det_feats_all[keep_idx]                       # (K,D)
            tr_feats = [t.get_feat() for t in strack_pool]
            rdist = _cosine_dist_matrix(tr_feats, det_feats_keep)          # (M,K)
            alpha1 = getattr(self, "reid_iou_fuse_alpha",
                     getattr(self.args, "reid_iou_alpha", 0.5))
            gate = getattr(self, "reid_gate",
                   getattr(self.args, "reid_gate", None))
            dists = _fuse_iou_reid(dists, rdist, alpha=alpha1, gate=gate)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            # feature for this detection
            feat = None
            if det_feats_all is not None:
                feat = det_feats_keep[idet]

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, feat=feat)
                self._assign_identity(track, feat)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, feat=feat)
                self._assign_identity(track, feat)
                refind_stracks.append(track)


        # ===========================================
        # Step 3: Second association (low-score dets)
        # ===========================================
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s)
                                 for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)

        # ---- IoU + ReID fusion (2nd) ----
        if (det_feats_all is not None) and (len(detections_second) > 0):
            det_feats_second = det_feats_all[second_idx]                   # (S,D)
            tr_feats = [t.get_feat() for t in r_tracked_stracks]
            rdist = _cosine_dist_matrix(tr_feats, det_feats_second)        # (R,S)
            alpha2 = getattr(self, "reid_iou_fuse_alpha_second",
                     getattr(self.args, "reid_iou_alpha_second", 0.3))
            gate = getattr(self, "reid_gate",
                    getattr(self.args, "reid_gate", None))
            dists = _fuse_iou_reid(dists, rdist, alpha=alpha2, gate=gate)

        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            feat = None
            if det_feats_all is not None:
                feat = det_feats_second[idet]

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, feat=feat)
                self._assign_identity(track, feat)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, feat=feat)
                self._assign_identity(track, feat)
                refind_stracks.append(track)

        # unmatched tracked → lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ==========================================
        # Deal with unconfirmed tracks (1st leftovers)
        # ==========================================
        detections_u = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections_u)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections_u)

        # ---- ReID fusion for unconfirmed ----
        if (det_feats_all is not None) and (len(detections_u) > 0):
            det_feats_keep = det_feats_all[keep_idx]                       # (K,D)
            u_idx = torch.as_tensor(
            u_detection,
            dtype=torch.long,
            device=det_feats_keep.device if torch.is_tensor(det_feats_keep)
                   else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            ).flatten()

            if not torch.is_tensor(det_feats_keep):
                det_feats_keep = torch.as_tensor(det_feats_keep, device=u_idx.device)

            det_feats_u = det_feats_keep.index_select(0, u_idx) 
            tr_feats = [t.get_feat() for t in unconfirmed]
            rdist = _cosine_dist_matrix(tr_feats, det_feats_u)             # (Utr,U)
            alpha2 = getattr(self, "reid_iou_fuse_alpha_second",
                     getattr(self.args, "reid_iou_alpha_second", 0.3))
            gate = getattr(self, "reid_gate",
                   getattr(self.args, "reid_gate", None))
            dists = _fuse_iou_reid(dists, rdist, alpha=alpha2, gate=gate)

        matches, u_unconfirmed, u_detection_left = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            feat = None
            if (det_feats_all is not None) and (len(u_detection) > 0):
                det_feats_keep = det_feats_all[keep_idx]
                feat = det_feats_keep[u_detection[idet]]
            unconfirmed[itracked].update(detections_u[idet], self.frame_id, feat=feat)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # =========================
        # Step 4: Init new stracks
        # =========================
        for inew in u_detection_left:
            track = detections_u[inew]  # NOTE: u_detection_left는 detections_u 기준 인덱스
            if track.score < self.det_thresh:
                continue
            feat = None
            if (det_feats_all is not None) and (len(u_detection) > 0):
                det_feats_keep = det_feats_all[keep_idx]
                feat = det_feats_keep[u_detection[inew]]
            track.activate(self.kalman_filter, self.frame_id, feat=feat)
            self._assign_identity(track, feat)
            activated_starcks.append(track)

        # =========================
        # Step 5: Update state
        # =========================
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ---- finalize pools ----
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
