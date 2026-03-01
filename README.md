# TrackWithReID
소스 코드: [ByteTrack](https://github.com/FoundationVision/ByteTrack), [TransReID](https://github.com/damo-cv/TransReID)
Modified from the original ByteTrack and TransReID repository.

## Introduction
범죄자가 사람이 많은 장소에 숨었을 경우 추적이 어려워짐
저장된 외형 정보를 통해 다수 안에서 빠르게 사람을 식별할 수 있도록 추적 모델과 ReID 모델을 통합
사용한 사전학습 모델은 MOT17(ByteTrack) & TransReID(ViT)이다.

## Features
ByteTrack의 update 함수에 TransReID 사전학습 모델을 추가하여 유사도가 기준치 이상일 경우 기존 id대신 저장된 폴더 명으로 출력
화면에서 나갔다가 다시 들어와도 동일
더 자세한 정보는 [블로그](https://blog.naver.com/deoduck92/224200304078) 참고

## Implementation
실행 명령어
```bash
python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --reid_weights "$REID_WEIGHTS" \
  --gallery "$GALLERY_DIR" \
  --path "$VIDEO_PATH" \
  --fp16 --fuse --save_result
'''

## License
This project integrates:
- ByteTrack (MIT License)
- TransReID (MIT License)

All rights belong to the original authors.
