# Meta-Netze

Meta-Netze provides configurable pipelines for gradient-based meta-learning, with a focus on the rapid deployment of photovoltaic (PV) power forecasting models.

The project investigates how forecasting models can adapt to new PV sites using limited local data, particularly through Model-Agnostic Meta-Learning (MAML) and Almost No Inner Loop (ANIL).

## Master's Thesis

This repository contains the research code accompanying the master's thesis:

**Meta-Learning for Rapid Deployment of PV Power Forecasting Models. A sample efficiency study**

- **Author:** Jorge Contreras Domingo
- **Institution:** Rühr-University of Bochum
- **Year:** 2026

The thesis studies the sample efficiency of meta-learning for PV power forecasting: how much site-specific data is needed to adapt a forecasting model to a new deployment.

**Thesis document:** TODO — add a link to the thesis PDF or university repository.

## Features

- Gradient-based meta-learning experiments using MAML and ANIL
- PV power forecasting and adaptation to new sites
- Sample-efficiency studies with limited adaptation data
- YAML-based experiment configuration
- Custom task and task-sampler interfaces
- PyTorch/PyTorch Lightning-based pipelines using learn2learn

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Jorcont/thesis.git
cd thesis
```

Create and activate a Conda environment with Python 3.10:

```bash
conda create --name meta-netze python=3.10
conda activate meta-netze
```

Install the dependencies:

```bash
pip install -r requirements.txt
conda install pyg -c pyg
```

For detailed environment setup and troubleshooting, see the
[Meta-Netze Wiki](https://github.com/Jorcont/thesis/wiki).

## Running an Experiment

Experiments are configured through YAML files. These specify the task, task sampler, model, loss function, algorithm, and experiment parameters.

Before running an experiment, update the dataset paths and configuration settings for your environment.

Example:

```bash
python -u main.py conf/yaml/best_config_lstm_maml_baseline_testing.yaml
```

Select the appropriate configuration for the experiment you want to run.

Detailed configuration instructions and SLURM execution examples are documented in the
[Meta-Netze Wiki](https://github.com/Jorcont/thesis/wiki).

## Data Availability

Large datasets, training logs, and generated experiment outputs are excluded from this repository.

Running the experiments requires the corresponding datasets and correctly configured data paths. Cloning this repository alone does not provide the complete experimental data.

See the Wiki for data requirements, preprocessing instructions, and the expected directory structure.

## Documentation

Detailed installation, configuration, usage, and development documentation is maintained in the GitHub Wiki:

[Meta-Netze Wiki](https://github.com/Jorcont/thesis/wiki)

The documentation covers:

- Environment setup and dependencies
- Data preparation and directory structure
- Experiment configuration
- Training, adaptation, and evaluation
- Execution on SLURM clusters
- Custom tasks, task samplers, models, and losses
- Known limitations and troubleshooting

## Citation

If you use this code or build on the associated research, please cite:

```bibtex
@mastersthesis{contrerasdomingo2026metanetze,
  author = {Contreras Domingo, Jorge},
  title  = {Meta-Learning for Rapid Deployment of {PV} Power Forecasting Models. A sample efficiency study},
  school = {Ruhr-Universit{\"a}t Bochum},
  year   = {2026},
  type   = {Master's Thesis}
}
```

## License

TODO — specify the applicable license and add or link the LICENSE file.

Retain any existing license and copyright notices from the original codebase. Third-party dependencies remain subject to their respective licenses.
