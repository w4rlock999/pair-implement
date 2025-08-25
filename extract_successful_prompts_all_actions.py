import json

def extract_rating_10_prompts():
    """
    Extract all prompts with rating 10 from the PAIR agentic results file.
    """
    file_path = 'data/agentic/PAIR_agentic_results_target_gpt_oss_20b_ALL_ACTIONS.json'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rating_10_prompts = []
        
        # Iterate through all experiments
        for experiment in data:
            experiment_id = experiment.get('experiment_id', 'Unknown')
            objective = experiment.get('objective', 'Unknown')
            
            # Check each category in results (Authority, Logic, Roleplay)
            results = experiment.get('result', {})
            for category, prompts in results.items():
                for prompt_data in prompts:
                    if prompt_data.get('Rating') == 10:
                        rating_10_prompts.append({
                            'action_label': experiment.get('action_label', ''),
                            'experiment_id': experiment_id,
                            'objective': objective,
                            'category': category,
                            'prompt': prompt_data.get('Prompt', ''),
                            'response': prompt_data.get('Response', ''),
                            'agent_level_rating': prompt_data.get('Rating'),
                            'agent_level_raw_judge_response': prompt_data.get('Raw_Judge_Response', ''),
                            'model_level_rating': '',
                            'model_level_raw_judge_response': '',
                        })
        
        return rating_10_prompts
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
if __name__ == "__main__":
    successful_prompts = extract_rating_10_prompts()
    print(f"Found {len(successful_prompts)} successful prompts.")
    
    output_file_path = 'data/agentic/PAIR_agentic_successful_prompts_gpt_oss_20b.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(successful_prompts, f, indent=4)
        
    print(f"Successful prompts saved to {output_file_path}")