# **One Stage Detection**

原型为人脸检测的训练框架，现在拓展为支持多源训练数据的单阶段目标检测框架，并融合了一些通用的trick

## Dependence

- pytorch => 1.6
- torchvision => 0.7.0



## Getting Started

1.配置数据源  [cfgs/dangercar_data.py](https://gitlab.deepglint.com/jaiweili/one-stage-detection/-/blob/master/cfgs/dangercar_data.py) 

2.配置训练细节 [cfgs/dangercar_solver.py](https://gitlab.deepglint.com/jaiweili/one-stage-detection/-/blob/master/cfgs/dangercar_solver.py)

3.训练模型

`python -m torch.distributed.launch --nproc_per_node=8 tools/train_ddp_ms.py cfgs/dangercar_solver.py`

4.查看训练log  

`tail -fn 100 log/[model_name].log`

