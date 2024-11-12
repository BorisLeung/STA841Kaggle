# STA841Kaggle

The repository for the Kaggle project of STA841.

## Setup

As of the time of creating this file, the python environment is set up as the following. You are encourage to follow it.

1. Set up conda environment

`conda create -n sta841kaggle python=3.10`

2. Activate the environment

`conda activate sta841kaggle`

3. Install the pip libraries

`python -m pip install -r requirements.txt`

## Optuna Dashboard

Optuna dashboard is a handy library for visualizing the hyperparameter search space. It should have already be installed if one follows the above setup guide.

To launch the dashboard, simply run the following in terminal:

```
optuna-dashboard sqlite:///<your-local-database>.db
```

A local server should then be hosted for one to browse.
