# Happy Code

## Modern Tools
- [Ruff](https://github.com/astral-sh/ruff)
- [uv](https://github.com/astral-sh/uv)


## Install
using conda:
```bash
conda create -n happy_code python=3.10
conda activate happy_code

pip install -r requirements.txt
```
or you can use [uv](https://github.com/astral-sh/uv):
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
# install wheel first
uv pip install wheel
uv pip install -r  requirements.txt
```

## Rules
1. place model weights to `model_repo/`
2. place data to `data/`


## Models
- [ ] DeepSeek-VL-7B
  - [x] SFT
  - [x] LoRA
  - [ ] text-only
- [ ] Mamba
  - [ ] Mamba-vision


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