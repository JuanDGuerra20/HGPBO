import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import re
import glob
import ast

def hp_plotting(scores, hp_name, hp_list):
    for j in range(len(scores)):
        eval_name, evaluation = scores[j]
        print(evaluation.shape)
        for i in range(len(hp_list)):
            plt.plot(range(nbr_query), evaluation[i][:nbr_query], label=f"Init {hp_list[i]}")
        plt.title(f"{eval_name} with varying {hp_name}")
        plt.xlabel("Query Number")
        plt.ylabel(f"{eval_name}")
        plt.legend()
        plt.savefig(
            f"synthetic3/lossless_efficient/data-2025-04-12/hp_analysis/{eval_name}_varying_{hp_name}.svg")
        plt.close()

    for eval_name, evaluation in scores:
        plt.plot(hp_list, evaluation[:, nbr_query - 1], label=eval_name)

    plt.title(f"End Model Scores for different evals at {nbr_query} Queries")
    plt.xlabel(f"{hp_name}")
    plt.ylabel(f"Performance")
    plt.legend()
    plt.savefig(f"synthetic3/lossless_efficient/data-2025-04-12/hp_analysis/final_scores_varying_{hp_name}.svg")
    plt.savefig(f"synthetic3/lossless_efficient/data-2025-04-12/hp_analysis/final_scores_varying_{hp_name}")

    plt.close()

def format_data(data, new_line=False):
    score = []
    seperated = data.split("]")

    for sep in seperated:
        tok = sep.replace(", [", "")
        tok = tok.replace(", dtype=torch.float64)", "")
        tok = tok.replace("tensor(", "")
        tok = tok.replace(" ", "")
        tok = tok.replace("[", "")

        numbers = []
        if len(tok) == 0:
            continue
        if new_line:
            for num in tok.split("\n"):
                numbers.append(float(num))
        else:
            for num in tok.split(","):
                numbers.append(float(num))

        tensor = torch.tensor(numbers, dtype=torch.float64)
        if tensor.shape[0] != nbr_query:
            print(tensor)
            print("=====================================")
            print(sep)
            print("=====================================")

            raise ValueError("Shape is wrong")
        score.append(tensor)
    return np.array(score)
if __name__ == "__main__":
    # Load the data

    files = glob.glob(f"synthetic3/lossless_efficient/data-2025-04-12/csv/kappa_*_gamma_10_nu_0_5_model_state_100_queries_eps_2_init_6_train_iter_10_repetitions_20_noise_0,5")

    parent_r2 = []
    child_1_r2_over = []
    child_2_r2_over = []
    explor = []
    exploit = []
    nbr_query = 100
    print(files)

    kappa_list = [2, 3, 4, 5, 6, 7, 8, 9]
    str_ordering = [str(i) for i in kappa_list]
    print(str_ordering)

    str_ordering = [float(i) for i in str_ordering]
    str_ordering = np.array(str_ordering)
    sorting = np.argsort(str_ordering)
    rand_init = np.array(str_ordering)[sorting]
    print(kappa_list)
    print(str_ordering[sorting])

    for file in files:
        df = pd.read_csv(file)

        data = df.values
        better_exploration_score = format_data(data[2,1])
        better_exploitation_score = format_data(data[3,1])
        r2 = np.fromstring(data[4,1].strip("[]"), sep=" ")
        child_1_r2 = np.fromstring(data[5,1].strip("[]"), sep=" ")
        child_2_r2 = np.fromstring(data[6,1].strip("[]"), sep=" ")


        parent_r2.append(r2[:nbr_query])
        child_1_r2_over.append(child_1_r2[:nbr_query])
        child_2_r2_over.append(child_2_r2[:nbr_query])
        explor.append(np.mean(better_exploration_score, axis=0)[:nbr_query])
        exploit.append(np.mean(better_exploitation_score, axis=0)[:nbr_query])
    scores = [["Parent_R2", np.array(parent_r2)[sorting]], ["Child_1_R2", np.array(child_1_r2_over)[sorting]],
                    ["Child_2_R2", np.array(child_2_r2_over)[sorting]], ["Exploration", np.array(explor)[sorting]],
                    ["Exploitation", np.array(exploit)[sorting]]]
    hp_plotting(scores, "kappa_list", kappa_list)