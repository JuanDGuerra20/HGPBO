import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import glob
import os
def hp_plotting(scores, hp_name, hp_list):
    '''for j in range(len(scores)):
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
        plt.close()'''

    for eval_name, evaluation in scores:
        plt.plot(hp_list, evaluation[:, nbr_query - 1], label=eval_name)

    plt.title(f"End Model Scores for different evals at {nbr_query} Queries")
    plt.xlabel(f"{hp_name}")
    plt.ylabel(f"Performance")
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.savefig(f"bad_modular_1_child/lossless_efficient/data-2025-05-26/hp_analysis/final_scores_varying_{hp_name}.svg")
    plt.savefig(f"bad_modular_1_child/lossless_efficient/data-2025-05-26/hp_analysis/final_scores_varying_{hp_name}")
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

def get_ordered_scores(file_list, hp, model, dataset):
    parent_r2 = []
    avg_child_r2 = []
    explor = []
    auc_over = []

    hp_vals = []
    for file in file_list:
        split = file.split("_")
        if model != "vanilla":
            loc = split.index(hp)
            val = split[loc+1]
        elif dataset == "synthetic3" and hp == "kappa":
            val = split[1]
        elif hp == "kappa":
            val = split[2]
        else:
            loc = split.index(hp)
            val = split[loc + 1]
        val = float(val.replace(",","."))
        hp_vals.append(val)

    combined = zip(hp_vals, file_list)
    combined = sorted(combined)
    hp_vals, sorted_files = zip(*combined)

    for file in sorted_files:
        df = pd.read_csv(file)
        val = df.values
        explor.append(float(val[1,1]))
        parent_r2.append(float(val[2,1]))
        if model == "vanilla":
            if '[' in val[3,1]:
                auc = val[3,1]
                auc = auc.strip('[]').replace('\n',' ')
                auc = np.fromstring(auc, dtype=float, sep=' ')
                auc_over.append(auc.sum())
            else:
                auc_over.append(float(val[3,1]))
        else:
            avg_child_r2.append(float(val[3,1]))
            if '[' in val[4,1]:
                auc = val[4,1]
                auc = auc.strip('[]').replace('\n',' ')
                auc = np.fromstring(auc, dtype=float, sep=' ')
                auc_over.append(auc.sum())
            else:
                auc_over.append(float(val[4,1]))

    if not os.path.exists(f'hp_plots/{model}/{dataset}'):
        os.mkdir(f'hp_plots/{model}/{dataset}')

    plt.plot(hp_vals, explor, label=f"RO")
    plt.plot(hp_vals, parent_r2, label="Parent R2")
    if model != "vanilla":
        plt.plot(hp_vals, avg_child_r2, label="Child R2")
    #plt.plot(hp_vals, auc_over, label="AUC")
    plt.legend()
    plt.title(f"HP Search for {hp} {model}")
    plt.ylim((-0.1, 1.1))
    plt.xlabel(f"{hp} value")
    plt.ylabel(f"Performance")
    plt.savefig(f'hp_plots/{model}/{dataset}/{hp}_{model}.png')
    plt.savefig(f'hp_plots/{model}/{dataset}/{hp}_{model}.svg')
    plt.close()
    print(f"\n==============================================================================================")
    print(f"Dataset : {dataset}\tModel : {model}")
    print(f"{hp} values: {hp_vals}")
    print(f"AUC: {auc_over}")
    best = np.argmax(auc_over)
    print(f"Best model: {hp_vals[best]} with AUC {auc_over[best]}")
    #return explor, parent_r2, avg_child_r2, auc_over, hp_vals



if __name__ == "__main__":
    # Load the data
    datasets = ['synthetic3', 'synthetic_tests', "sub_2D", "modular_2D"]
    models = ["lossless_efficient", "laferriere_model", "vanilla"]
    for d in datasets:
        for m in models:
            if m == "vanilla":
                files = glob.glob(f"{d}/{m}/data-2025-11-18/csv/kappa_*_model_state_100_queries_init_1_train_iter_10_repetitions_*")
                #get_ordered_scores(files, "kappa", m, d)
            else:
                files = glob.glob(f"{d}/{m}/data-2025-11-19/csv/*_rep_init_3_train_iter_10_eps_*_k_7,5_g_*_nu_0_5_noise_0,1.csv")
                get_ordered_scores(files, "g", m, d)