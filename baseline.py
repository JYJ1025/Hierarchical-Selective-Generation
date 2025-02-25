import json
import math
import os
import csv
import openai
import matplotlib.pyplot as plt

client = openai.OpenAI(api_key = '')

def language_generator(
    messages, 
    model="gpt-3.5-turbo",   
    max_tokens=500,               
    temperature=0,           
    stop=None,
    seed=123,      
    tools=None,
    logprobs=True,  
):
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "logprobs": logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion

def entailment_oracle(seq1, seq2):
    checking_prompt = f"""
        You are an expert in Natural Language Inference. Your task is to determine whether the first sentence contradicts the second sentence.

        Please provide an answer with a score of 0 or 1, where:
        0 = contradiction
        1 = no contradiction

        The first sentence: {seq1}
        The second sentence: {seq2}

        Provide your score only with number.
        Score:
    """
    result = language_generator(
        [{"role": "user", "content": checking_prompt}],
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0
    )
    score = result.choices[0].message.content.strip()
    return int(score) if score.isdigit() else 0

def final_completion():
    with open("data/data1.json", "r") as f:
        data = json.load(f)
    
    baseline_results = []
    
    for question_data in data:        
        question = question_data["question"]
        answer = question_data.get("transformed")
    
        # Step 1: generate initial response
        API_RESPONSE = language_generator(
            [{"role": "user", "content": question}],
            model="gpt-4o-mini", 
            logprobs=True,
        )
        response_text = API_RESPONSE.choices[0].message.content.strip()
    
        # Step 2: get initial score
        logprobs = [token.logprob for token in API_RESPONSE.choices[0].logprobs.content]
        sum_of_logprobs = sum(logprobs)
        average_logprob = sum_of_logprobs / len(logprobs)
        initial_prob = math.exp(average_logprob) * 100
    
        # Step 3: access entailment oracle
        is_contradict = entailment_oracle(response_text, answer)
        
        is_IDK = None
        
        result = {
            "question": question_data["question"],
            "response_text": response_text,
            "current_score": initial_prob,
            "is_IDK": is_IDK,
            "is_contradict": is_contradict,
            "answer": answer
        }
        
        baseline_results.append(result)
        
    output_path = "data1_cache_/data1_response_0.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(baseline_results, outfile, indent=4, ensure_ascii=False)

def plot_current_score_distribution(json_file_path):
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    current_scores = [item["current_score"] for item in data]

    plt.figure(figsize=(10, 6))
    plt.hist(current_scores, bins=20, edgecolor='k', alpha=0.7)
    plt.title("Distribution of Current Scores")
    plt.xlabel("Current Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
def main():
    final_completion()
    # plot_current_score_distribution ("data1_cache_/data1_response_0.json")

if __name__ == "__main__":
    main()
