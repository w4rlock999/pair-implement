import json

# Open and read the JSON file
with open('data/agentic/PAIR_agentic_results_target_gpt_oss_20b_ALL_ACTIONS_success_rate_with_info.json', 'r') as file:
    data = json.load(file)

# Open and read the actions_with_risk.json file
with open('data/agentic/actions_with_risk.json', 'r') as file:
    actions_data_list = json.load(file)

# Convert actions_data from a list to a dictionary for easier lookup
actions_data = {item['label']: item for item in actions_data_list}


# Modify each entry according to the requirements
for entry in data:
    # Rename action_risk to static_ASR
    if 'action_risk' in entry:
        entry['static_ASR'] = entry.pop('action_risk')
    
    # Rename success_rate_percentage to PAIR_ASR and convert percentage to decimal
    if 'success_rate_percentage' in entry:
        entry['PAIR_ASR'] = entry.pop('success_rate_percentage') / 100
    
    # Rename total_experiments to total_PAIR_experiments
    if 'total_experiments' in entry:
        entry['total_PAIR_experiments'] = entry.pop('total_experiments')
    
    # Rename successful_experiments to successful_PAIR_experiments
    if 'successful_experiments' in entry:
        entry['successful_PAIR_experiments'] = entry.pop('successful_experiments')
    
    # Add context_last_message based on actions_with_risk.json
    action_label = entry.get('action_label')
    if action_label and action_label in actions_data:
        action_info = actions_data[action_label]
        if 'input' in action_info and action_info['input']:
            last_message = action_info['input'][-1]
            message_type = last_message.get('type')
            if message_type == 'tool':
                tool_name = last_message.get('name', '')
                entry['context_last_message'] = message_type
                entry['tool_name'] = tool_name
            else:
                entry['context_last_message'] = message_type
                entry['tool_name'] = ''
        
        # Add input token from actions_with_risk.json
        if 'input' in action_info and isinstance(action_info['input'], list):
            for message in reversed(action_info['input']):
                if message.get('type') == 'ai':
                    response_metadata = message.get('response_metadata')
                    if response_metadata:
                        token_usage = response_metadata.get('token_usage')
                        if token_usage:
                            prompt_tokens = token_usage.get('prompt_tokens')
                            if prompt_tokens is not None:
                                entry['input token'] = prompt_tokens
                                break  # Stop after finding the first one from the end
                    # Check usage_metadata as a fallback
                    usage_metadata = message.get('usage_metadata')
                    if usage_metadata:
                         prompt_tokens = usage_metadata.get('input_tokens')
                         if prompt_tokens is not None:
                            entry['input token'] = prompt_tokens
                            break


# Save the modified data back to the file
with open('data/agentic/PAIR_agentic_results_target_gpt_oss_20b_ALL_ACTIONS_success_rate_with_info.json', 'w') as file:
    json.dump(data, file, indent=2)

# Print the modified data
print(json.dumps(data, indent=2))
