Advanced Application in Artificial Vision
Overview
This repository contains a Jupyter Notebook titled "Advanced Application in Finances". This notebook covers advanced techniques and applications in the field of finances, including API conections to fetch in real time the stock prices and forecasting tools

Contents
-Data fetching
-Data exploration
-Data preprocessing
-Forecasting
-Forecasting evaluation

Prerequisites
To run the notebook, you need to have the following installed:

Python 3
Jupyter Notebook or JupyterLab
Importing Necessary Python libraries:

import yfinance as yf
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from random import sample
from datetime import timedelta
from itertools import cycle

from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt.plots import plot_objective, plot_convergence
from sklearn.ensemble import GradientBoostingRegressor
