import os
import sys
import json
import math
import random
import numpy as np
import pickle
import glob
import argparse
import warnings
import scipy.stats as st
from scipy import stats
from scipy.stats import beta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def u_binom(k, n, delta=0.05):
    hi = stats.beta.ppf(1 - delta, k+1, n-k)
    hi = 1.0 if math.isnan(hi) else hi

    return hi

def save_dict_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def find_optimal_threshold(question_dict, target_error_rate, delta=0.05):
    all_scores = sorted([data["current_score"] for q in question_dict.values() for data in q.values()])
    left, right = 0, len(all_scores) - 1
    optimal_threshold = None
    iteration_count = 0  

    while left <= right:
        iteration_count += 1
        mid = (left + right) // 2
        threshold = all_scores[mid]

        total_above_threshold = 0
        contradict_above_threshold = 0

        for question, iterations in question_dict.items():
            for iteration, data in iterations.items():
                score = data["current_score"]
                is_contradict = data["is_contradict"]

                if score > threshold:
                    total_above_threshold += 1
                    if is_contradict == 0:
                        contradict_above_threshold += 1
                    break  

        error_rate = u_binom(contradict_above_threshold, total_above_threshold, delta)

        if error_rate <= target_error_rate:
            if optimal_threshold is None or threshold < optimal_threshold:
                optimal_threshold = threshold
            right = mid - 1
        else:
            left = mid + 1

    if optimal_threshold is not None:
        # print(f"\n Optimal Threshold Found: {optimal_threshold} (after {iteration_count} iterations)\n")
        return optimal_threshold
    else:
        # print("\n Failed to find an optimal threshold. No value meets the target error rate.\n")
        return None

# def box_plot_results(val_ratios, contradict_ratios, iteration_avgs, target_error_rate, none_count, save_path):
#     plt.figure(figsize=(12, 7))

#     # Selectivity (none IDK ratio)
#     plt.subplot(3, 1, 1)
#     plt.boxplot(val_ratios, vert=True, patch_artist=True)
#     plt.ylabel("Selectivity")
#     plt.title(f"Ratio of Questions Above Threshold")

#     # FDR-E (contradict_ratios)
#     plt.subplot(3, 1, 2)
#     plt.boxplot(contradict_ratios, vert=True, patch_artist=True)
#     plt.axhline(y=target_error_rate, color='gray', linestyle='dashed', label=f'Target Error Rate: {target_error_rate}')
#     plt.ylabel("FDR-E")
#     plt.title(f"Contradiction Ratio Above Threshold\n(Target Error Rate: {target_error_rate}, None Count: {none_count})")
#     plt.legend()

#     # Selection Efficiency (iteration_avgs)
#     plt.subplot(3, 1, 3)
#     plt.boxplot(iteration_avgs, vert=True, patch_artist=True)
#     plt.ylabel("Selection Efficiency")
#     plt.title(f"Total Iterations to Exceed Threshold")
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     # plt.show()
    
def box_plot_results(val_ratios, contradict_ratios, iteration_avgs, target_error_rate, none_count, save_path):
    plt.figure(figsize=(12, 7))

    # Selectivity (none IDK ratio)
    plt.subplot(3, 1, 1)
    bp1 = plt.boxplot(val_ratios, vert=True, patch_artist=True)
    plt.ylabel("Selectivity")
    plt.title("Ratio of Questions Above Threshold")

    mean_val = np.mean(val_ratios)
    se_val = st.sem(val_ratios)
    ci_val = st.t.interval(0.95, len(val_ratios)-1, loc=mean_val, scale=se_val)
    
    plt.errorbar(1, mean_val, yerr=[[mean_val - ci_val[0]], [ci_val[1] - mean_val]], fmt='o', color='black', label="95% CI")
    plt.legend()

    # FDR-E (contradict_ratios)
    plt.subplot(3, 1, 2)
    bp2 = plt.boxplot(contradict_ratios, vert=True, patch_artist=True)
    plt.axhline(y=target_error_rate, color='gray', linestyle='dashed', label=f'Target Error Rate: {target_error_rate}')
    plt.ylabel("FDR-E")
    plt.title(f"Contradiction Ratio Above Threshold\n(Target Error Rate: {target_error_rate}, None Count: {none_count})")
    
    mean_contr = np.mean(contradict_ratios)
    se_contr = st.sem(contradict_ratios)
    ci_contr = st.t.interval(0.95, len(contradict_ratios)-1, loc=mean_contr, scale=se_contr)
    plt.errorbar(1, mean_contr, yerr=[[mean_contr - ci_contr[0]], [ci_contr[1] - mean_contr]], fmt='o', color='black', label="95% CI")
    plt.legend()

    # Selection Efficiency (iteration_avgs)
    plt.subplot(3, 1, 3)
    bp3 = plt.boxplot(iteration_avgs, vert=True, patch_artist=True)
    plt.ylabel("Selection Efficiency")
    plt.title("Total Iterations to Exceed Threshold")
    
    mean_iter = np.mean(iteration_avgs)
    se_iter = st.sem(iteration_avgs)
    ci_iter = st.t.interval(0.95, len(iteration_avgs)-1, loc=mean_iter, scale=se_iter)
    plt.errorbar(1, mean_iter, yerr=[[mean_iter - ci_iter[0]], [ci_iter[1] - mean_iter]], fmt='o', color='black', label="95% CI")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def cross_validate(question_dict, target_error_rate, save_path, num_splits, delta=0.05):
    questions = list(question_dict.keys())
    val_ratios = []
    contradict_ratios = []
    iteration_avgs = []
    none_count = 0
    
    for i in range(num_splits):
        random.shuffle(questions)
        train_questions = set(questions[:4000])
        val_questions = set(questions[4000:])
        
        train_dict = {q: question_dict[q] for q in train_questions}
        val_dict = {q: question_dict[q] for q in val_questions}

        # output_dir = "data1_cache_/cross_validation_results"
        # os.makedirs(output_dir, exist_ok=True)  
        # train_filename = os.path.join(output_dir, f"train_dict_trial_{i+1}.json")
        # val_filename = os.path.join(output_dir, f"val_dict_trial_{i+1}.json")

        # save_dict_to_json(train_dict, train_filename)
        # save_dict_to_json(val_dict, val_filename)
        
        optimal_threshold = find_optimal_threshold(train_dict, target_error_rate, delta)
        
        # if optimal threshold doesn't exist
        if optimal_threshold is None:
            none_count += 1
            print(f"Trial {i+1}: Optimal threshold not found. Assigning default value (e.g., 0).")
            continue
        
        print(f"Trial {i+1}: Optimal threshold = {optimal_threshold}")

        total_above_threshold = 0
        contradict_above_threshold = 0
        iteration_sum = 0 
        
        for question, iterations in val_dict.items():
            iteration_count = 0 
            for iteration, data in iterations.items():
                if data["current_score"] > optimal_threshold:
                    total_above_threshold += 1
                    iteration_sum += iteration_count  
                    if data["is_contradict"] == 0:
                        contradict_above_threshold += 1
                    break 
                iteration_count += 1
        
        val_ratio = total_above_threshold / len(val_dict) if len(val_dict) > 0 else 0
        contradict_ratio = contradict_above_threshold / total_above_threshold if total_above_threshold > 0 else 0
        iteration_avg = iteration_sum / total_above_threshold if total_above_threshold > 0 else 0
        
        val_ratios.append(val_ratio)
        contradict_ratios.append(contradict_ratio)
        iteration_avgs.append(iteration_avg)
        
        print(f"   Validation Set - Ratio of questions above threshold: {val_ratio:.4f}")
        print(f"   Validation Set - Contradict ratio above threshold: {contradict_ratio:.4f}")
        print(f"   Validation Set -  Average iteration to reach threshold: {iteration_avg:.2f}")
        
    box_plot_results(val_ratios, contradict_ratios, iteration_avgs, target_error_rate, none_count, save_path)      
        
with open("data1_cache_/question_dict.json", "r", encoding="utf-8") as f:
    question_dict = json.load(f)

output_dir = "test_result"
os.makedirs(output_dir, exist_ok=True)

for target_error_rate in np.arange(0.2, 0.33, 0.01):
    save_path = os.path.join(output_dir, f"result_target_{target_error_rate:.2f}.png")
    print(f"\n=== Running test for target error Rate: {target_error_rate:.2f} ===")
    cross_validate(question_dict, target_error_rate, save_path, num_splits=100, delta=0.05)
    

