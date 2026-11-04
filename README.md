# Meta-Netze
This project aims to provide pipelines for meta-learning and meta-Reinforcement-Learning (meta-RL) respectively. It mainly focusses on task distributions related to energy grids. However, utilizing the pipelines for other task distributions only requires for minor changes, namely child classes for Task and TaskSampler classes.

## Getting started
To setup the project first clone the repository, then create a virtual environment with python version 3.10 and activate it. 
```bash
conda create --name meta-netze python=3.10
conda activate meta-netze
```

Then install the dependencies from the `requirements.txt` file and install PyG.
```bash
pip install -r requirements.txt
conda install pyg -c pyg
```

## Running the project
To start a meta-learning experiment for a particular use case:
Create a config yaml file to configure the pipeline, Task, TaskSampler, Loss function and Model objects.
The possible config options can be found in the config folder.

- Task and TaskSampler:
Configs can be found in "task_config.py".
Specification is given by their name (as a string), e.g. "Grid" or "HPRegression".
Each use case requires for a use case specific Task and Tasksampler class, e.g. the GridTask and the GridTaskSampler provided by this Repo.

- Loss function:
Configs can be found in "loss_config.py".

- Model:
Default Models can be found in the "model" folder. However, you are always welcome to include your own model. Just make sure, that forward and backward processing of your model works with the functions used in this Repo. 
Note: This Repo only supports Pytorch/Pytorch Lightning.
Note: Model configuration is not debuged yet.

- Algorithm:
This Repo is ment to be algorithm-agnostic.
However, yet the meta-learning pipeline is based on MAML with learn2learn (l2l) package.
Later versions will have a meta-learning pipeline plus meta-algo specific child classes in "algorithm" folder.

## Custom Task and TaskSampler
To create your own Task and TaskSampler classes:
1. Name them as MYSpecificNameTask and MySpecificNameTaskSampler and inherit them from TaskSampler and SupervisedLearningTask (RLTask for meta-RL).
2. Customize task-specific functions like import_data, split_data or make_task so that data generation, data spliting and task generation are adjusted accordingly.
Note: Functions like fast_adapt or run_testing are not likely to require changes, but you should still check them.
3. Make sure that losses are returned accordingly, so that meta-model updates can be performed. This might need specific consideration of your model. However, the learn2learn (l2l) package should cover these problems for you.
