# Florence-2-base 模型引用

用途
- Microsoft Florence-2 系列基础模型，用于开放词表目标检测/区域描述等任务，生成候选框与标签。

约定
- 本目录不存放实际权重，`source` 为指向 Hugging Face 缓存目录的符号链接。
- 当前链接示例：
  `source -> /home/igame/.cache/huggingface/hub/models--microsoft--Florence-2-base`

重新链接（示例）
```bash
rm -rf source
ln -s "$HOME/.cache/huggingface/hub/models--microsoft--Florence-2-base" source
```

提示
- 首次使用前请确保已通过 `transformers` 或 `huggingface_hub` 成功下载到本机缓存。
