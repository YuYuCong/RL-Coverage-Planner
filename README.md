# Terrain-aware Complete Coverage Path Planning using Deep Reinforcement Learning
This repository applies deep reinforcement learning to the complete coverage path planning problem.
The aim for the project is to research if a RL-agent can be trained to take terrain information into account.
That is, to see whether a RL-agent will adapt its path for complete-coverage planning when the agent is presented with terrain-information.
This will enable the agent to be more energy-efficient than classical planning methods.

## Installation
After cloning the repository, create a conda environment using the `environment.yml file`.

```
conda env create -f environment.yml
conda activate RL
```

These commands should install all the needed dependencies.

## Run the code

### Environment Dashboard
This script will start dashboard on your local computer.  
The dashboard lets you play around with the different parameters used to create environments.

```
python3 env_dashboard.py
```

Then open a browser and go to [localhost:8050](http://127.0.0.1:8050/).

### Train DQN-agent

You can train an agent using the following command.
This command will run with all the default parameters.

```
python3 train.py --savePath="<folder>"
```

This will start the training process and save all the results in the `<repository>/data/` folder.
The following arguments can be added to the run command in order to change the default settings:

* `--loadArguments=` :
  * `folder,name`
  * folder and name of a text file that can be used to load arguments
  * 参数文件txt
* `--heightRequired` :
  * indicator whether terrain information should be used or not
  * 是否使用地形信息
* `--dim=` :
  * `<dim_x>,<dim_y>`
  * default: 16,16
  * dimension of the environment in x and y direction
  * 环境维度尺寸
* `--hFreq=` :
  * `<hFreq_x>,<hFreq_y>`
  * default: 2,2
  * frequency of variation of the terrain (higher values means more variation, the dimension should be a multiple of the frequency)
  * 地形变化的频率（值越高意味着变化越大，环境维度尺寸应该是频率的倍数）
* `--oFreq=` :
  * `<oFreq_x>,<oFreq_y>`
  * default: 2,2
  * frequency of the variation of the obstacles (higher values means more variation, the dimension should be a multiple of the frequency)
  * 障碍物变化的频率（值越高表示变化越大，环境维度尺寸应为频率的倍数）
* `--fillRatio=` :
  * `<ratio>`
  * default: 0.14
  * initial ratio used to guide obstacle construction (the obstacle ratio will not exact match this fill ratio)
  * 用于指导障碍物构建的初始比率（障碍物比率大概等于此填充比率）
* `--loadEnv=` :
  * `<folder>,<name>`
  * folder and name of a file that stores an environment representation
  * 存储环境表示的文件夹和文件的名称
* `--movePunish=` :
  * `<punishment>`
  * default: 0.05
  * the punishment given to the agent when moving
  * 移动惩罚
* `--terrainPunish=` :
  * `<punishment>`
  * default: 0.05
  * the punishment given to the agent for the terrain difference
  * 由于地形差异而给予智能体的惩罚
* `--obstaclePunish=` :
  * `<punishment>`
  * default: 0.5
  * the punishment given to the agent when colliding with an obstacle
  * 碰撞惩罚
* `--discoverReward=` :
  * `<reward>`
  * default: 1.0
  * the reward given to the agent when the agent discovers a new tile
  * 发现新图块的奖励
* `--coverageReward=` :
  * `<reward>`
  * default: 50.0
  * the reward given tot he agent when the agent has covered all tiles
  * 全覆盖的最终奖励
* `--maxStepMultiplier=` :
  * `<multiplier>`
  * default: 2
  * related to the maximum number of steps an agent can do before a run will be ended by the environment
  * 与环境结束运行之前代理可以执行的最大步骤数相关
* `--networkGen=` :
  * options: \[simpleQ, simpleQ2\]
  * default: simpleQ
  * indicates the type of network the agent will use
  * 网络结果的类型
* `--rlAgent=` :
  * options: \[deepQ, doubleDQ\]
  * default: doubleDQ
  * indicates the type of agent that will be used
  * 智能体的类型
* `--gamma=` :
  * `<value>`
  * default: 0.9
  * the gamma value used during training, this value indicates how much value is given to future rewards
  * 未来奖励相关参数
* `--epsilonDecay=` :
  * `<value>`
  * default: 2000
  * indicates how rapidly the epsilon values decays, epsilon is used to balance between exploration and exploitation
  * 表示 epsilon 值衰减的速度，epsilon 用于平衡勘探和开发
* `--targetUpdate=` :
  * `<value>`
  * default: 1000
  * indicates how frequently the target network is updated with the policy network
  * 表示目标网络更新策略网络的频率
* `--nbEpisodes` :
  * `<value>`
  * default: 2000
  * number of episodes the agent is trained on
  * 训练轮次数量
* `--printEvery=` :
  * `<value>`
  * default: 50
  * indicates how frequently results are printed to the command line
  * 每隔多少轮在终端输出一次结果
* `--saveEvery=` :
  * `<value>`
  * default: 250
  * indicates how frequently results are saved
  * 每隔多少轮保存一次测试结果
* `--softmax=` :
  * options:  \[True, False\]
  * default: False
  * indicates whether softmax action selection is used over epsilon-greedy action selection
  * softmax是否使用
* `--savePath=` :
  * `<folder>`
  * folder where the results are saved, should be provided!
  * 存储路径

### Visualise training results

The following command renders the training result.
It will display a window that shows the agent interacting in the environment.
While during training, an epsilon-greedy policy is used, the agent acts in this visualization according to a completely greedy policy.

```
python3 visualize.py --loadTrainArgs=<repository>/results/8x_multi/ --episodeNb=30000 --visDim=800,800
```

The following arguments can be added to the run command in order to change the default settings:

* `--loadTrainArgs=` :
  * `<folder>`
  * folder where the arguments and models are saved, should be provided!
* `--episodeNb=` :
  * `<value>`
  * default: 250
  * episode number that will be loaded
* `--visDim=` :
  * `<dim_x>,<dim_y>`
  * default: 500,500
  * the dimension of the visualization
* `--fps=` :
  * `<value>`
  * default: 2
  * the frames per second of the visualization


After running the command, you should see something like below:
[![First Results](https://cdn.loom.com/sessions/thumbnails/25ebefb4e32f4aa2af2014cb5bf990ac-with-play.gif)](https://www.loom.com/share/25ebefb4e32f4aa2af2014cb5bf990ac)