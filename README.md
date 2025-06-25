# PRSnet

## Install

```bash
torch==2.0.1
tensorboard==2.4.1
tensorflow==2.4.1
pip install scipy tqdm torchsummary
```

## Data

shapenet: https://hyper.ai/cn/datasets/16769

## Training

```py
python train.py --config ./configs/default_config.yaml
```

## Test

```python
python test.py --config ./configs/default_config.yaml
```

