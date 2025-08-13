"""
Adapated from Guay-Hottin et al. (2025)
Original github: https://github.com/RoseGH20/apiBO/tree/main
"""


import torch
import pandas as pd
from scipy.stats import multivariate_normal
from typing import Dict, List
import numpy as np

def get_prior_mean(all_dist_normed: torch.Tensor, prior_cond: str):
    flat = torch.flatten(all_dist_normed)
    if prior_cond == "good":
        idx = torch.argsort(flat)
        dist = flat[idx]
    elif prior_cond == "misleading":
        idx = torch.argsort(flat)[::-1]
        dist = flat[idx]
    return idx, dist


def get_prior_std(X_test: torch.Tensor, prior_std_perc: float = 0.10):
    min_val = X_test.min()
    max_val = X_test.max()
    std = (max_val - min_val) * prior_std_perc
    std = std.numpy()
    return std


def create_prior(prior_mean: torch.Tensor, prior_std: torch.Tensor, X_test: torch.Tensor):
    rv = multivariate_normal(mean=prior_mean.detach().numpy(), cov=prior_std.flatten().detach().numpy())
    disc_prior = rv.pdf(torch.arange(len(X_test)))
    return disc_prior


def get_grouped_df(df: pd.DataFrame, dataset_name: str):
    if dataset_name == "AgNP":
        grouped_df = df.groupby(["QAgNO3(%)", "Qpva(%)", "Qtsc(%)", "Qseed(%)", "Qtot(uL/min)"])[
            "loss"].mean().reset_index()
        grouped_df.rename(columns={"loss": "gt_mean"}, inplace=True)
    elif dataset_name == "AutoAM":
        grouped_df = df.groupby(["Prime Delay", "Print Speed", "X Offset Correction", "Y Offset Correction"])[
            "Score"].mean().reset_index()
        grouped_df.rename(columns={"Score": "gt_mean"}, inplace=True)
    elif dataset_name == "CrossedBarrel":
        grouped_df = df.groupby(["n", "theta", "r", "t"])["toughness"].mean().reset_index()
        grouped_df.rename(columns={"toughness": "gt_mean"}, inplace=True)
    return grouped_df


def check_config(config: Dict):
    valid_dataset_name = ["AgNP", "AutoAM", "CrossedBarrel"]
    valid_strategy = ["vbo", "pbi", "apibo", "pibo"]
    assert config["dataset"][
               "name"] in valid_dataset_name, "Invalid dataset name, please select a dataset from {}.".format(
        valid_dataset_name)

    for strat in config["strategy"]:
        strat in valid_strategy, "Invalid strategy name {}, please select a strategy from {}.".format(strat,
                                                                                                      valid_strategy)

    if "apibo" in config["strategy"]:
        assert "alpha" in config.keys(), "Please specify 'alpha' for apibo."

    if "pibo" in config["strategy"] and "beta" not in config.keys():
        config["beta"] = min(config["dataset"]["n_iters"] / 10, 5)
        print("A value was not specified for beta, setting it to min(n_iters/10, 5) = {}.".format(config["beta"]))

    if "pibo" in config["strategy"] or "apibo" in config["strategy"]:
        if "prior_std" not in config.keys():
            config["prior_std"] = 0.1
            print("A value was not specified for the standard deviation of the prior, setting it to 0.1.")

    if "nrnd" not in config.keys():
        config["nrnd"] = 3
        print("Using default number of initial random points, nrnd = 3.")

    if "nrep" not in config.keys():
        config["nrep"] = 20
        print("Using default number of repetitions, nrep = 20.")