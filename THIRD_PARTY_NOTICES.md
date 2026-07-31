# Third-Party Notices

## Segment Anything 2 (SAM 2)

Dance Focus uses [facebookresearch/sam2](https://github.com/facebookresearch/sam2),
pinned to commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

Copyright (c) Meta Platforms, Inc. and affiliates.

SAM 2 source code and model checkpoints are distributed under the Apache License
2.0. The complete license is available at:

https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE

The SAM 2.1 Small checkpoint is downloaded directly from Meta's official public
model host on first use and is verified with its published SHA-256 digest.

## Torchreid OSNet-AIN

Dance Focus includes a minimal OSNet-AIN model definition adapted from
[KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid),
pinned to commit `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`.

Copyright (c) 2018 Kaiyang Zhou. Distributed under the MIT License. The complete
license is included at `LICENSES/TORCHREID-MIT.txt`.

The OSNet-AIN MSMT17 checkpoint is downloaded from the maintainer's official
Hugging Face repository at revision `a5c5cc037c24235cda3b21085b93ad77c9616224`
and verified with SHA-256. ReID features remain local and are not persisted.

## Torchvision Keypoint R-CNN

Dance Focus uses torchvision's Keypoint R-CNN COCO_V1 model for pose-aware
framing. Torchvision is distributed under the BSD 3-Clause License. Pretrained
model use may also be affected by the terms of its training datasets; users are
responsible for reviewing those terms for their deployment.
