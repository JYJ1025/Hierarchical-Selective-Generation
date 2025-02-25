import json
import math
import os
import csv
import openai

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

def semantic_tree_walker(answer):
    simplification_prompt = f"""
        You are an expert in Natural Language Inference.
        Your task is to rewrite the following complex sentence in order to make it easier to understand by non-native speakers.
        You can do so by replacing complex words with simpler synonyms (i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long complex sentence into several simpler ones. 
        The final simplified sentence needs to be grammatical, fluent, and retain the main ideas of its original counterpart without altering its meaning.
        complex: {answer}    
        
        Generalized sentence:    
    """ 
    
    # Step 1: get abstracted response
    entailed_response = language_generator(
        [{"role": "user", "content": simplification_prompt}],
        model="gpt-3.5-turbo",
        logprobs=True,
        max_tokens=100,  
    )  
    response_text = entailed_response.choices[0].message.content.strip()
    
    # Step 2: get entail probability (for update score)
    logprobs = [token.logprob for token in entailed_response.choices[0].logprobs.content]
    sum_of_logprobs = sum(logprobs)
    average_logprob = sum_of_logprobs / len(logprobs)
    entail_prob = math.exp(average_logprob) * 100
    
    # print("[Semantic Tree Walker]")   
    # print(f"Response Text: {response_text}")
    # print(f"Token Count: {len(logprobs)}") 
    # print(f"Entailment Probability: {entail_prob}")

    return response_text, entail_prob

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

def final_completion(start_index=0, max_entailments=30, output_dir="data1_cache_"):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: get QA data
    with open('data/data1.json', 'r', encoding='utf-8') as f:
        ground_data = json.load(f)

    answers = {item["question"]: item["transformed"] for item in ground_data if "question" in item and "transformed" in item}

    # Step 2: start tree walker
    for i in range(max_entailments):
        current_file = os.path.join(output_dir, f"data1_response_{start_index + i}.json")
        
        with open(current_file, "r", encoding="utf-8") as infile:
            data = json.load(infile)
            
        updated_data = []
        
        for item in data:
            # Step 2-1: get previous data
            question = item.get("question", "")
            response_text = item.get("response_text", "")
            current_score = item.get("current_score", 0)
            is_IDK = item.get("is_IDK", "")
            answer = answers.get(question, "")
            is_contradict = entailment_oracle(response_text, answer)
            
            # Step 2-2: get response data
            new_response_text, entail_probability = semantic_tree_walker(response_text)
            current_score += entail_probability
            
            # Step 2-3: access entailment oracle
            is_contradict = entailment_oracle(new_response_text, answer)
            
            updated_item = {
                "question": question,
                "response_text": new_response_text,
                "current_score": current_score,
                "is_IDK": is_IDK,
                "is_contradict": is_contradict,
                "answer": answer
            }
            updated_data.append(updated_item)
        
        next_file = f"data1_response_{start_index + i + 1}.json"
        output_file = os.path.join(output_dir, next_file)
        
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(updated_data, outfile, indent=4, ensure_ascii=False)
        
        print(f"[Iteration {i}] Processed '{current_file}' -> Saved '{next_file}'")
        

def main():
    output_dir = "data1_cache_"
    max_entailments = 30
    start_index = 0
    final_completion(start_index, max_entailments, output_dir)

if __name__ == "__main__":
    main()
    