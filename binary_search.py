import json
import math
from scipy import stats
from scipy.stats import beta

with open("data1_cache_/question_dict.json", "r", encoding="utf-8") as f:
    question_dict = json.load(f)

def u_binom(k, n, delta=0.05):
    hi = stats.beta.ppf(1 - delta, k+1, n-k)
    hi = 1.0 if math.isnan(hi) else hi

    return hi

def find_optimal_threshold(question_dict, target_error_rate, delta=0.05):
    all_scores = sorted([data["current_score"] for q in question_dict.values() for data in q.values()])
    left, right = 0, len(all_scores) - 1
    optimal_threshold = None
    iteration_count = 0  

    print("\n Start Binary Search for Optimal Threshold...\n")

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
        
        print(f"Iteration {iteration_count}:")
        print(f"   ➤ Threshold: {threshold}")
        print(f"   ➤ Total Above Threshold: {total_above_threshold}")
        print(f"   ➤ Contradictions Above Threshold: {contradict_above_threshold}")
        print(f"   ➤ Error Rate: {error_rate:.4f} (Target: {target_error_rate:.4f})")
        print("-" * 50)

        if error_rate <= target_error_rate:
            if optimal_threshold is None or threshold < optimal_threshold:
                optimal_threshold = threshold
            right = mid - 1
        else:
            left = mid + 1

    if optimal_threshold is not None:
        print(f"\n Optimal Threshold Found: {optimal_threshold} (after {iteration_count} iterations)\n")
        return optimal_threshold
    else:
        print("\n Failed to find an optimal threshold. No value meets the target error rate.\n")
        return None

target_error_rate = 0.26
optimal_threshold = find_optimal_threshold(question_dict, target_error_rate)
