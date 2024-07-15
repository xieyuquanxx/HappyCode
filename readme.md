# Happy Code
DPO for MLLM.

## Modern Tools
- [Ruff](https://github.com/astral-sh/ruff)
- [uv](https://github.com/astral-sh/uv)


## Install for DPO
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv .venv/.dpo
source .venv/.dpo/bin/activate
uv pip install -r  requirements.txt
uv pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# install minerl from source
uv pip install env/minerl --no-cache-dir --no-build-isolation
# install flash-attention2
uv pip install flash-attn --no-build-isolation
# clean cache
uv clean
```

## Rules
1. place model weights to `model_repo/`
2. place data to `data/`


## DPO for Models
对于每一个模型，都用`uv`为其创建一个虚拟环境，例如:`uv venv .deepseek`，`uv venv .mamba`，这样不同模型依赖不会相互破坏。
- [x] [DeepSeek-VL-7B](https://github.com/deepseek-ai/DeepSeek-VL)


## Tips
### Soft Link
模型权重可能在`/data1/models/xxx`，但是项目在`/data2/project/yyy`，为了不去重复保存模型权重，可以使用`soft link`的方式
```bash
ln -s <模型权重> <软连接的目录>
```

例如，`deepseek`权重在`/data1/Models/deepseek-vl-7b-chat`，本目录在`/data/Users/xyq/developer/happy_code`，项目的模型权重在`model_repo`下，那么命令为：
```bash
ln -s /data1/Models/deepseek-vl-7b-chat /data/Users/xyq/developer/happy_code/model_repo
```