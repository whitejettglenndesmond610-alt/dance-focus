# Dance Focus

Dance Focus 是一个基于 Meta [Segment Anything 2](https://github.com/facebookresearch/sam2) 的 Linux 舞蹈视频自动重构工具。在第一帧框选自己后，SAM 2 会传播目标人物的分割掩码，应用再生成平滑裁剪轨迹，让目标尽量保持在输出画面中央。

> 当前为 Linux GPU 原型，已支持指定人物追踪、动态裁剪预览和高画质 GPU 导出。

## 开源基础

- 核心追踪：`facebookresearch/sam2`，约 1.96 万 GitHub stars，Apache-2.0
- 固定源码提交：`2b90b9f5ceec907a1c18123530e92e794ad901a4`
- 官方模型：SAM 2.1 Small，184 MB
- 视频播放：PySide6 Qt Multimedia
- 视频解码与帧缓存：OpenCV
- 视频编码和原音频复用：FFmpeg

选择 SAM 2 而不是普通框追踪器，是因为它原生接受人物框提示，并利用视频记忆传播同一目标的像素级掩码。详细第三方归属见 `THIRD_PARTY_NOTICES.md`。

## 当前功能

- 在第一帧拖框指定要跟随的舞者
- SAM 2.1 视频目标分割和遮挡记忆
- Keypoint R-CNN 多人姿态分析，以肩髋躯干作为构图锚点
- OSNet-AIN 人物外观校验，辅助遮挡后身份找回
- 240 帧分段处理和 8 帧掩码重叠续接，避免长视频一次占满内存
- 9:16、16:9、1:1、4:5 输出画幅
- 离线平滑裁剪路径，减少镜头抖动
- 根据人物远近自动推近和拉远，同时保持固定输出分辨率
- 红橙置信度时间轴，以及上一异常、下一异常快速导航
- 位置、缩放和自动跟随强度镜头关键帧
- 人物修正关键帧，只重新跟踪当前修正点到下一个修正点
- 稳定、平衡、灵敏三档镜头稳定，以及自动速度和加速度限制
- 撤销、重做和可取消的后台分析
- 项目原子自动保存、schema 1/2兼容迁移和手动打开项目
- 播放、暂停、时间轴和原音频预览
- 原始画面与直接读取源视频的高质量构图预览切换
- 清新明亮的分阶段工作台，以及可折叠的高级镜头调整
- 任意帧添加人工修正框后重新传播
- 导出时直接解码源视频，避免JPEG分析缓存造成二次画质损失
- RTX NVENC高画质编码，自动回退到可用的软件编码器
- 原生、720p、1080p输出，以及源帧率、30 FPS、60 FPS选项
- 可选60 FPS运动插帧，并保持视频与音频时长一致
- MP4导出并以192 kbps AAC保留源视频音频

## 系统要求

- Linux
- Python 3.12，由 `uv` 自动管理
- FFmpeg
- 推荐 NVIDIA GPU；当前已在 RTX 5070 Ti 12 GB 上验证
- CPU 可以运行，但 SAM 2 视频推理会很慢

## 安装运行

```bash
git clone https://github.com/whitejettglenndesmond610-alt/dance-focus.git
cd dance-focus
SAM2_BUILD_CUDA=0 uv sync
uv run dance-focus
```

如需关闭界面动效，可以设置：

```bash
DANCE_FOCUS_REDUCE_MOTION=1 uv run dance-focus
```

首次运行智能分析时会下载三套官方权重：

```text
~/.cache/dance-focus/models/sam2.1_hiera_small.pt
~/.cache/dance-focus/models/osnet_ain_x1_0_msmt17.pth
~/.cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth
```

SAM 2.1 Small 约 184 MB，OSNet-AIN 约 16.5 MB，Keypoint R-CNN 约 226 MB。下载完成后不会重复下载；SAM 2和OSNet权重会额外校验SHA-256。解码后的JPEG帧缓存位于 `~/.cache/dance-focus/videos/`。

项目自动保存在 `~/.local/state/dance-focus/projects/`。应用重新启动时保持空白，可通过“打开项目”手动恢复。项目文件包含提示框、逐帧跟踪状态、姿态锚点、构图参数、镜头关键帧、镜头路径和导出设置，但不保存ReID人物特征。

分析和预览会使用JPEG帧缓存；最终导出始终重新读取源视频，以避免缓存压缩损失。系统必须能够通过OpenCV解码源视频。运行错误会记录到 `~/.cache/dance-focus/dance-focus.log`，也可在界面点击“运行日志”。

## 使用方法

1. 点击“打开视频”。
2. 在第一帧拖动鼠标，完整框住自己，尽量不要框入旁边的人。
3. 选择输出画幅、镜头稳定模式和自动缩放范围，点击“运行 SAM 2 跟踪”。
4. SAM、姿态与ReID分析完成后会自动切换到构图预览，点击“播放构图”检查动态裁剪。
5. 如需检查人物追踪框，在“预览模式”切回“原视频 + 裁剪框”。
6. 使用时间轴红橙标记或“上一异常/下一异常”检查风险帧。
7. 如需人工构图，在当前帧设置位置、缩放和跟随强度后写入镜头关键帧。
8. 如果遮挡后仍跟错，点击“修正当前帧人物”并重新框选。应用只会重算该帧到下一个人物修正点。
9. 在“导出”阶段选择画质、分辨率和帧率。低帧率素材导出60 FPS时可按需开启流畅插帧。
10. 点击“导出跟拍成片”，完成后可立即打开经过解码验证的结果文件。

构图预览直接读取源视频画面，并按源视频帧率实时播放。选择的1080p和60 FPS运动插帧属于最终导出规格，会在编码阶段生成；界面顶部会同时显示预览帧率和导出帧率。

## 验证

```bash
uv run pytest
```

当前测试覆盖裁剪几何、SAM掩码边界、局部区间拼接、项目迁移、撤销重做、分段帧映射和导出编码参数。项目还完成了以下端到端验证：

- 合成移动目标：60/60 帧有效追踪
- 真实群舞素材：1280×720 HEVC、903 帧、30.10 秒
- 裁剪结果：404×720 H.264、903 帧、AAC原音频、30.10 秒
- Qt裁剪预览、后台导出状态和导出文件重新播放
- RTX NVENC 1280×720、60 FPS运动插帧，视频与AAC音频均保持1.00秒

## 故障排查

- 播放或导出异常时，点击界面中的“查看运行日志”。
- 日志位置：`~/.cache/dance-focus/dance-focus.log`
- 模型位置：`~/.cache/dance-focus/models/`
- 视频帧缓存：`~/.cache/dance-focus/videos/`
- 删除某个视频对应的帧缓存后重新追踪，可以强制重新解码该视频。

## 限制

SAM 2 是视频对象分割模型。应用已叠加OSNet-AIN身份校验和短时位移门控，但多人穿着相似并发生长时间完全遮挡、人物离开画面后重新进入或镜头切换时，仍可能切换目标；此时应添加人物修正关键帧。
