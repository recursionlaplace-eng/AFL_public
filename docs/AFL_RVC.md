# AFL RVC Convert

`AFL RVC Convert` 是一个给 ComfyUI 用的 **RVC 推理节点**。

它现在已经按便携包思路整理成下面这种结构：

- AFL 节点代码在 `custom_nodes/AFL_public`
- RVC 推理核心仓库内置在 `custom_nodes/AFL_public/vendor/rvc_repo`
- RVC 目标音色模型统一放在 `ComfyUI/models/tts/RVC`

它不是零样本克隆节点。  
它的作用是：

- 输入一段原始音频
- 选择一个 **已经训练好的 RVC 音色模型**
- 输出转换后的音频

---

## 节点用途

适合：

- 已经有 RVC `.pth` 模型的说话/唱歌音色转换
- 想把 RVC 作为 AFL 便携包的一部分直接发给别人

不适合：

- 仅靠一段参考音频临时克隆音色
- 纯零样本音色克隆

---

## 节点输入

### 必填

- `audio`
- `model_name`
- `pitch_shift`
- `f0_method`
- `index_rate`
- `protect`

### 选填

- `rvc_repo_path`
- `model_path`
- `index_path`
- `filter_radius`
- `resample_sr`
- `rms_mix_rate`
- `device`
- `is_half`

---

## 现在推荐的使用方式

### 1. 把模型放到这里

把 RVC 模型统一放到：

`E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\models\tts\RVC`

例如：

`E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\models\tts\RVC\default_voice\default.pth`

`E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\models\tts\RVC\default_voice\added_IVF511_Flat_nprobe_1_default_v2.index`

节点会自动扫描这个目录下的 `.pth`，并在 `model_name` 里显示出来。

### 2. 正常情况不用手填路径

如果你已经把模型放在上面的目录里：

- `model_name` 直接下拉选
- `model_path` 留空
- `index_path` 留空

节点会自动：

- 找到对应的 `.pth`
- 尝试自动配对同目录下的 `.index`

### 3. 只有特殊情况才用手填

如果你选 `model_name = __manual__`，才会使用：

- `model_path`
- `index_path`

---

## 便携包分发说明

如果你要把整个 ComfyUI 便携包发给别人，别人 **不需要再单独安装 RVC**，前提是你发出去的包里已经包含：

### 必须带上的内容

1. `custom_nodes/AFL_public`
2. `custom_nodes/AFL_public/vendor/rvc_repo`
3. `ComfyUI/models/tts/RVC`
4. `python_embeded` 里已经装好的依赖

### 当前这个 AFL RVC 节点依赖

见：

`E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\custom_nodes\AFL_public\requirements-rvc-wrapper.txt`

主要包括：

- `ffmpeg-python`
- `faiss-cpu`
- `pyworld`
- `torchcrepe`
- `praat-parselmouth`
- `fairseq`

### RVC 资产文件

还需要保留：

- `E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\custom_nodes\AFL_public\vendor\rvc_repo\assets\hubert\hubert_base.pt`
- `E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\custom_nodes\AFL_public\vendor\rvc_repo\assets\rmvpe\rmvpe.pt`

---

## 默认建议参数

- `pitch_shift = 0`
- `f0_method = rmvpe`
- `index_rate = 0.66`
- `protect = 0.33`

---

## 参数理解

### `pitch_shift`

整体升降调，单位是半音。

- `0` 不改
- `+12` 升八度
- `-12` 降八度

### `f0_method`

音高提取方式。

推荐先用：

- `rmvpe`

### `index_rate`

RVC 检索特征参与强度。

- 高一点：更像目标模型
- 低一点：更保留原始发音结构

### `protect`

保护辅音、气声、咬字的程度。

- 太低容易糊
- 太高可能音色融合不够

---

## 当前已放入的样例模型

当前便携包里已经放了一个样例模型：

- `E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\models\tts\RVC\default_voice\default.pth`
- `E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\models\tts\RVC\default_voice\added_IVF511_Flat_nprobe_1_default_v2.index`

体积大约：

- `.pth` 约 `55.2 MB`
- `.index` 约 `63.0 MB`
- 合计约 `118.2 MB`

这个样例主要用来：

- 验证节点能跑
- 让便携包开箱即用

---

## 注意事项

### 1. 修改节点后要重启 ComfyUI 后台

只刷新前端不够。  
Python 节点改动后，要 **完整重启 ComfyUI**。

### 2. 这不是零样本克隆

这个节点必须依赖现成 RVC 模型。

### 3. 唱歌效果主要取决于模型本身

节点只是推理入口。  
如果模型本身更偏说话，唱歌效果也不会自动变强。

---

## 故障排查

### 下拉里没有模型

检查目录：

`E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\models\tts\RVC`

并确认里面真的有 `.pth` 文件。

### 选了模型但报模型不存在

通常是：

- 文件被移动了
- 路径缓存还没刷新
- ComfyUI 没重启

### 报 RVC 依赖错误

说明 `python_embeded` 里的依赖没装完整，或便携包缺文件。

### 报 hubert / rmvpe 相关错误

检查这两个文件是否还在：

- `E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\custom_nodes\AFL_public\vendor\rvc_repo\assets\hubert\hubert_base.pt`
- `E:\comfyui\ComfyUI_Mie_2026_V8.0\ComfyUI\custom_nodes\AFL_public\vendor\rvc_repo\assets\rmvpe\rmvpe.pt`

