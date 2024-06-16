# Happy Code

## Modern Tools
- [Ruff](https://github.com/astral-sh/ruff)
- [uv](https://github.com/astral-sh/uv)


## Install for GPT2
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv .gpt2 --python <python3.10 path>
source .gpt2/bin/activate

uv pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt

```

## Rules
1. place model weights to `model_repo/`
2. place data to `data/`


## Models
对于每一个模型，都用`uv`为其创建一个虚拟环境，例如:`uv venv .deepseek`，`uv venv .mamba`，这样不同模型依赖不会相互破坏。
- [ ] [DeepSeek-VL-7B](https://github.com/deepseek-ai/DeepSeek-VL)
  - [x] SFT
  - [x] LoRA
  - [ ] text-only
  - [x] [flash-attention2](https://github.com/Dao-AILab/flash-attention)
- [ ] [Mamba](https://github.com/state-spaces/mamba)
  - [x] Mamba
  - [x] Mamba2
  

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