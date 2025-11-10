# dnn_models 目录说明

本目录用于存放“模型资源”的本地引用，建议通过符号链接（symlink）指向已下载到 Hugging Face 缓存中的模型，以避免将大体积权重纳入仓库。

约定
- 每个子目录代表一个模型系列（例如 `qwen-2.5-vl/`、`florence-2-base/`）。
- 子目录下的 `source` 为指向真实模型目录的符号链接（只读引用）。
- 模型权重不直接提交到 Git，仅保持链接与最小化说明文档。

示例（已创建）
- `dnn_models/qwen-2.5-vl/source -> /home/igame/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct`
- `dnn_models/florence-2-base/source -> /home/igame/.cache/huggingface/hub/models--microsoft--Florence-2-base`

快速操作
```bash
# 创建（或重建）符号链接
ln -s /path/to/hf_cache/models--Org--Model dnn_models/<model-name>/source
```

注意
- 如果换了机器或用户，请根据实际 HF 缓存路径重新创建 `source` 链接。
- 需要忽略整个 `dnn_models/` 或其中的链接时，可在仓库根 `.gitignore` 里添加相应规则。
