# AFL 音频节点显存策略

这个插件会被 AFL Canvas 通过 ComfyUI workflow/API 连续调用。这里的首要目标不是极限速度，而是让不同工作流连续执行时不容易因为上一个音频节点残留显存而卡死或爆显存。

## 当前策略

下面这些 AFL 重音频节点会主动处理显存释放：

- `AFL Seed VC`
- `AFL Audio Separate Demucs 4 Stem`
- `AFL Audio Separate Voice Background`
- `AFL Audio Separate Demucs 6 Stem Beta`
- `AFL Audio Qwen Subtitles`

这些节点在加载大模型前，会先请求 ComfyUI 释放可卸载的 GPU 显存。

`AFL Seed VC` 采用折中策略：运行结束后把模型搬回 CPU 并清理 CUDA 缓存，显存不长期占用；如果短时间内再次调用 SeedVC，会把 CPU 里的模型搬回 GPU，避免每次都从硬盘完整重载。SeedVC 空闲 5 分钟后会自动彻底卸载，也可以手动使用 `AFL Unload Seed VC`。

音频分离和 Qwen 字幕节点采用更稳的策略：运行结束后断开模型引用并清理缓存，不长期保留模型对象。

这个策略接近 LongCat `keep_model_loaded=true` 的方向，但 SeedVC 额外加了空闲超时全卸载。AFL Canvas 经常连续调用不同类型的工作流，所以这里优先避免插件自己偷偷长期占用显存。

## 代价

音频分离和字幕节点下次再运行时，可能会慢一点，因为模型需要重新加载。

SeedVC 连续调用时通常不会每次从硬盘完整重载，但仍会发生 CPU/GPU 之间的搬运。这是为了降低“上一个音频工作流留下显存，导致下一个图片/视频/音频工作流失败或卡住”的概率。

## 轻量节点

只做张量或文本处理的节点，例如响度匹配、stem 混音、文本显示，不会长期加载大模型，不需要同级别的清理策略。

## 如果还不够稳

可以继续缩短音频长度、减小分块、换小模型，或者让最重的音频步骤走 CPU。CPU 会慢，但比显存爆掉更适合连续 API 工作流。
