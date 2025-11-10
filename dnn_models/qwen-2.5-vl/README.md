# Qwen2.5-VL（7B Instruct）模型引用

用途
- 多模态视觉语言模型（VLM），用于对候选主体进行命名与简介生成（可输出结构化 JSON）。

约定
- 本目录不存放实际权重，`source` 为指向 Hugging Face 缓存目录的符号链接。
- 当前链接示例（固定到快照 hash）：
  `source -> $HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<hash>`

重新链接（示例）
```bash
rm -rf source
HASH=$(cat "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/refs/main")
ln -s "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/$HASH" source
```

提示
- 首次使用前请确保已通过 `transformers` 或 `huggingface_hub` 成功下载到本机缓存。
