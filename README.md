# MoTwithReID

## Introduction
Tracking AI는 단순히 사람이 어디있는지 bounding box를 친다.
치매 노인분들을 위해 갤러리에 저장된 사람의 경우 tracking_id 대신에 저장된 이름으로 출력하도록 한다.
이를 위해 ReID 모델과 Tracking 모델을 멀티모달로 구현했다.

## BackBone 모델
Tracking: ByteTrack
ReID: TransReID
