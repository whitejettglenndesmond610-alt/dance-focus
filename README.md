# Dance Focus

Dance Focus 是一个基于 Meta [Segment Anything 2](https://github.com/facebookresearch/sam2) 的 Linux 舞蹈视频自动重构工具。在第一帧框选自己后，SAM 2 会传播目标人物的分割掩码，应用再生成平滑裁剪轨迹，让目标尽量保持在输出画面中央。

> 当前为 Linux GPU 原型，已支持指定人物追踪、动态裁剪预览和带原音频导出。

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
- 240 帧分段处理和 8 帧掩码重叠续接，避免长视频一次占满内存
- 9:16、16:9、1:1、4:5 输出画幅
- 离线平滑裁剪路径，减少镜头抖动
- 播放、暂停、时间轴和原音频预览
- 原视频裁剪框与铺满播放器的最终成片预览切换
- 任意帧添加人工修正框后重新传播
- MP4 导出并保留源视频音频

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

首次点击“使用 SAM 2 自动追踪”时，应用会从 Meta 官方地址下载 SAM 2.1 Small 权重到：

```text
~/.cache/dance-focus/models/sam2.1_hiera_small.pt
```

下载完成后会校验官方 SHA-256，后续不会重复下载。解码后的 JPEG 帧缓存位于 `~/.cache/dance-focus/videos/`。

播放和导出会优先读取该帧缓存，因此不依赖系统是否支持源视频的 HEVC/H.265 解码。运行错误会记录到 `~/.cache/dance-focus/dance-focus.log`，也可在界面点击“查看运行日志”。

## 使用方法

1. 点击“打开视频”。
2. 在第一帧拖动鼠标，完整框住自己，尽量不要框入旁边的人。
3. 选择输出画幅，点击“使用 SAM 2 自动追踪”。
4. 追踪完成后会自动切换到“裁剪结果（成片）”，点击“播放”或按空格键检查动态裁剪视频。
5. 如需检查人物追踪框，在“预览模式”切回“原视频 + 裁剪框”。
6. 如果遮挡后跟错，暂停到对应帧，重新框选自己，再运行一次追踪。
7. 点击“导出视频”。
8. 导出完成后点击“打开导出视频”播放经过可解码验证的结果文件。

## 验证

```bash
uv run pytest
```

当前测试覆盖裁剪几何、SAM掩码边界和分段帧映射。项目还完成了以下端到端验证：

- 合成移动目标：60/60 帧有效追踪
- 真实群舞素材：1280×720 HEVC、903 帧、30.10 秒
- 裁剪结果：404×720 H.264、903 帧、AAC原音频、30.10 秒
- Qt裁剪预览、后台导出状态和导出文件重新播放

## 故障排查

- 播放或导出异常时，点击界面中的“查看运行日志”。
- 日志位置：`~/.cache/dance-focus/dance-focus.log`
- 模型位置：`~/.cache/dance-focus/models/`
- 视频帧缓存：`~/.cache/dance-focus/videos/`
- 删除某个视频对应的帧缓存后重新追踪，可以强制重新解码该视频。

## 限制

SAM 2 是视频对象分割模型，不是人物身份识别模型。多人穿着相似并发生长时间完全遮挡、人物离开画面后重新进入或镜头切换时，仍可能切换目标；此时需要添加人工修正框。后续如需自动找回身份，应叠加 Apache/MIT 许可的人体检测和 ReID 模型。
