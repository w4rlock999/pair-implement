from openai import OpenAI, RateLimitError
import os   
import time
import json
from dotenv import load_dotenv
import concurrent.futures
import traceback

load_dotenv()

# Groq client
groq_client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                     base_url="https://api.groq.com/openai/v1"
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROQ_TARGET_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "gpt-4o-mini"

with open('judge_prompt.txt', 'r', encoding='utf-8') as f:
    judge_system_prompt = f.read()

def call_judge_llm(prompt, response, objective):
    """
    Calls the judge LLM (OpenAI GPT-4o-mini) to evaluate the response.
    Handles rate limit errors with exponential backoff.
    """
    # Construct the full prompt with objective, prompt, and response
    judge_prompt = f"Jailbreaking Prompt: {prompt}\n Model's Response: {response}"
    
    messages = [
        {"role": "system", "content": judge_system_prompt.replace(r"[OBJECTIVE]", objective)},
        {"role": "user", "content": judge_prompt}
    ]
    
    retries = 5
    delay = 1.0
    for i in range(retries):
        try:
            # Always use OpenAI client for judge
            result = openai_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                max_tokens=1000
            )
            return result.choices[0].message.content
        except RateLimitError:
            print(f"Rate limit reached for judge LLM. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{retries})")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"An unexpected error occurred in call_judge_llm for prompt '{prompt[:50]}...': {e}")
            traceback.print_exc()
            return f"Error: {e}"
    
    print("Max retries reached for judge LLM. Skipping.")
    return "Error: Max retries reached for judge LLM."


def call_target_llm(prompt):
    """
    Calls the target LLM (Groq gpt-oss-20b) and returns the response.
    Handles rate limit errors with exponential backoff.
    """
    retries = 5
    delay = 1.0
    for i in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_TARGET_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except RateLimitError:
            print(f"Rate limit reached for target LLM. Retrying in {delay:.2f} seconds... (Attempt {i + 1}/{retries})")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"An unexpected error occurred in call_target_llm for prompt '{prompt[:50]}...': {e}")
            traceback.print_exc()
            return f"Error: {e}"

    print("Max retries reached for target LLM. Skipping.")
    return "Error: Max retries reached for target LLM."

def process_prompt(prompt_info):
    """
    Processes a single prompt: gets response from target model and then gets judge rating.
    """
    try:
        new_response = call_target_llm(prompt_info['prompt'])
        
        if new_response.startswith("Error:"):
            new_judge_response = "N/A"
            new_rating = None
        else:
            new_judge_response = call_judge_llm(
                prompt_info['prompt'], 
                new_response, 
                prompt_info['objective']
            )
            
            new_rating = None
            if not new_judge_response.startswith("Error:") and "[[" in new_judge_response and "]]" in new_judge_response:
                try:
                    start = new_judge_response.find("[[") + 2
                    end = new_judge_response.find("]]")
                    new_rating = int(new_judge_response[start:end])
                except (ValueError, IndexError):
                    new_rating = None
        
        return {
            "action_label": prompt_info['action_label'],
            "experiment_id": prompt_info['experiment_id'],
            "objective": prompt_info['objective'],
            "category": prompt_info['category'],
            "prompt": prompt_info['prompt'],
            "agent_level_response": prompt_info.get('response') or prompt_info.get('agent_level_response'),
            "agent_level_rating": prompt_info['agent_level_rating'],
            "agent_level_raw_judge_response": prompt_info.get('raw_judge_response') or prompt_info.get('agent_level_raw_judge_response'),
            "model_level_response": new_response,
            "model_level_rating": new_rating,
            "model_level_raw_judge_response": new_judge_response
        }
    except Exception as e:
        print(f"Error processing prompt for experiment {prompt_info.get('experiment_id', 'N/A')}: {e}")
        traceback.print_exc()
        return None

def inject_and_evaluate_jailbreak_prompts_to_standalone_model(jailbreak_prompts, max_workers=10):
    """
    Re-evaluates the jailbreak prompts with the new ratings concurrently.
    """
    results = []
    
    print(f"Re-evaluating {len(jailbreak_prompts)} prompts with rating 10 using up to {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {executor.submit(process_prompt, prompt_info): prompt_info for prompt_info in jailbreak_prompts}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_prompt), 1):
            prompt_info = future_to_prompt[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                print(f"Processed prompt {i}/{len(jailbreak_prompts)} (Experiment ID: {prompt_info['experiment_id']})")
            except Exception as exc:
                print(f"Prompt from experiment {prompt_info['experiment_id']} generated an exception: {exc}")
                traceback.print_exc()

    output_file = 'data/agentic/PAIR_agentic_successful_prompts_gpt_oss_20b_with_model_ratings.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
        print(f"Successfully processed {len(results)} prompts")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return results

if __name__ == "__main__":
    # Extract all prompts with rating 10
    with open('data/agentic/PAIR_agentic_successful_prompts_gpt_oss_20b.json', 'r', encoding='utf-8') as f:
        jailbreak_prompts = json.load(f)
    
    # Display summary of found prompts
    for i, prompt_info in enumerate(jailbreak_prompts, 1):
        print(f"=== Prompt {i} ===")
        print(f"Action Label: {prompt_info['action_label']}")
        print(f"Experiment ID: {prompt_info['experiment_id']}")
        print(f"Objective: {prompt_info['objective'][:100]}..." if len(prompt_info['objective']) > 100 else f"Objective: {prompt_info['objective']}")
        print(f"Category: {prompt_info['category']}")
        print(f"Original Rating: {prompt_info['agent_level_rating']}")
        print(f"Prompt: {prompt_info['prompt'][:150]}..." if len(prompt_info['prompt']) > 150 else f"Prompt: {prompt_info['prompt']}")
        print("-" * 80)
    
    print("\n" + "="*80)
    
    # Automatically proceed with re-evaluation if prompts are found
    if jailbreak_prompts:
        print(f"\nProceeding with re-evaluation of {len(jailbreak_prompts)} prompts...")
        results = inject_and_evaluate_jailbreak_prompts_to_standalone_model(jailbreak_prompts)
        
        # Display summary of results
        if results:
            print(f"\n=== SUMMARY ===")
            print(f"Total prompts processed: {len(results)}")
            ratings_summary = {}
            for result in results:
                new_rating = result['model_level_rating']
                if new_rating is not None:
                    ratings_summary[new_rating] = ratings_summary.get(new_rating, 0) + 1
                else:
                    ratings_summary['Unknown'] = ratings_summary.get('Unknown', 0) + 1
            
            print("New rating distribution:")
            # Sort ratings, putting 'Unknown' at the end
            sorted_ratings = []
            unknown_count = 0
            
            for rating, count in ratings_summary.items():
                if rating == 'Unknown':
                    unknown_count = count
                else:
                    sorted_ratings.append((rating, count))
            
            # Sort numeric ratings
            sorted_ratings.sort(key=lambda x: x[0])
            
            # Print numeric ratings first
            for rating, count in sorted_ratings:
                print(f"  Rating {rating}: {count} prompts")
            
            # Print unknown ratings last
            if unknown_count > 0:
                print(f"  Rating Unknown: {unknown_count} prompts")
    else:
        print("No prompts with rating 10 found.")