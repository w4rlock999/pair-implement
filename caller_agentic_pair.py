#!/usr/bin/env python3
"""
Caller Script for PAIR Algorithm with Agentic Mode

This script demonstrates how to use the enhanced PAIR algorithm that supports both 
standard and agentic modes for red teaming AI agents. It loads objectives from 
failed_objective_with_exp_output.json and agentic contexts from top_bottom_actions.json.

The script supports configurable parameters for:
- Number of objectives to process
- Number of actions to use as context  
- Start indices for both objectives and actions
- PAIR algorithm parameters

Usage:
    python caller_agentic_pair.py --max_objectives 5 --max_actions 3 --start_obj 0 --start_action 0
"""

import json
import argparse
import os
import sys
from datetime import datetime
from pair_groq_gpt_oss_20b_agentic import pair_algorithm

def load_objectives(file_path):
    """Load objectives from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            objectives = json.load(f)
        print(f"Loaded {len(objectives)} objectives from {file_path}")
        return objectives
    except FileNotFoundError:
        print(f"Error: Objectives file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in objectives file: {e}")
        return []

def load_actions(file_path):
    """Load actions from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            actions = json.load(f)
        print(f"Loaded {len(actions)} actions from {file_path}")
        return actions
    except FileNotFoundError:
        print(f"Error: Actions file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in actions file: {e}")
        return []

def extract_agentic_context(action_data):
    """
    Extract and format agentic context from action data.
    
    Args:
        action_data: Dictionary containing action information with 'input' field
        
    Returns:
        str: Formatted context string for the agentic PAIR algorithm
    """
    if 'input' not in action_data:
        return ""
    
    context_parts = []
    
    # Add action label and risk score
    label = action_data.get('label', 'unknown')
    risk = action_data.get('risk', 0.0)
    context_parts.append(f"Action: {label} (Risk Score: {risk})")
    
    # Process input messages
    for i, message in enumerate(action_data['input']):
        msg_type = message.get('type', 'unknown')
        content = message.get('content', '')
        
        # Truncate very long content to keep context manageable
        if len(content) > 1000:
            content = content[:1000] + "... [truncated]"
        
        context_parts.append(f"\nMessage {i+1} ({msg_type}):")
        context_parts.append(content)
        
        # Add tool calls if present
        if 'additional_kwargs' in message and 'tool_calls' in message['additional_kwargs']:
            tool_calls = message['additional_kwargs']['tool_calls']
            if tool_calls:
                context_parts.append("Tool calls:")
                for j, tool_call in enumerate(tool_calls[:3]):  # Limit to first 3 tool calls
                    function_name = tool_call.get('function', {}).get('name', 'unknown')
                    context_parts.append(f"  - {function_name}")
    
    return "\n".join(context_parts)

def run_pair_experiment(objective_data, agentic_context, experiment_id, 
                       num_iterations=3, use_groq_attacker=False):
    """
    Run a single PAIR experiment with the given objective and context.
    
    Args:
        objective_data: Dictionary with 'objective' and 'expected_output' keys
        agentic_context: String containing the agent context history
        experiment_id: Unique identifier for this experiment
        num_iterations: Number of PAIR iterations to run
        use_groq_attacker: Whether to use Groq or OpenAI for the attacker
        
    Returns:
        dict: Results from the PAIR algorithm
    """
    print("\n" + "="*120)
    print(f"EXPERIMENT {experiment_id}")
    print("="*120)
    
    objective = objective_data['objective']
    expected_output = objective_data['expected_output']
    
    print(f"Objective: {objective}")
    print(f"Expected Output: {expected_output}")
    print(f"Context Length: {len(agentic_context)} characters")
    
    # Standard start string for jailbreaks
    start_string = "Sure, here's"
    
    try:
        # Run PAIR algorithm in agentic mode
        result = pair_algorithm(
            num_iterations=num_iterations,
            attack_objective=objective,
            start_string=start_string,
            use_groq_attacker=use_groq_attacker,
            agentic_context=agentic_context
        )
        
        return {
            'experiment_id': experiment_id,
            'objective': objective,
            'expected_output': expected_output,
            'agentic_context_length': len(agentic_context),
            'result': result,
            'success': any(
                any(attempt.get('Rating', 0) == 10 for attempt in approach_results) 
                for approach_results in result.values()
            )
        }
        
    except Exception as e:
        print(f"Error in experiment {experiment_id}: {str(e)}")
        return {
            'experiment_id': experiment_id,
            'objective': objective,
            'error': str(e)
        }

def save_results(results, output_file):
    """Save experiment results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run PAIR algorithm with agentic mode')
    
    # Data parameters
    parser.add_argument('--max_objectives', type=int, default=5,
                       help='Maximum number of objectives to process (default: 5)')
    parser.add_argument('--max_actions', type=int, default=3,
                       help='Maximum number of actions to use as context (default: 3)')
    parser.add_argument('--start_obj', type=int, default=0,
                       help='Start index for objectives (default: 0)')
    parser.add_argument('--start_action', type=int, default=0,
                       help='Start index for actions (default: 0)')
    
    # PAIR algorithm parameters
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of PAIR iterations per experiment (default: 3)')
    parser.add_argument('--use_groq_attacker', action='store_true',
                       help='Use Groq model for attacker (default: OpenAI GPT-4o-mini)')
    parser.add_argument('--mode', choices=['standard', 'agentic', 'both'], default='agentic',
                       help='Mode to run: standard PAIR, agentic PAIR, or both (default: agentic)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (default: auto-generated timestamp)')
    
    args = parser.parse_args()
    
    # Validate environment variables
    required_env_vars = ['OPENAI_API_KEY', 'GROQ_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    # Load data files
    objectives_file = 'data/agentic/failed_objective_with_exp_output.json'
    actions_file = 'data/agentic/top_bottom_actions.json'
    
    objectives = load_objectives(objectives_file)
    actions = load_actions(actions_file)
    
    if not objectives:
        print("No objectives loaded. Exiting.")
        sys.exit(1)
    
    if not actions and args.mode in ['agentic', 'both']:
        print("No actions loaded but agentic mode requested. Exiting.")
        sys.exit(1)
    
    # Slice data based on parameters
    end_obj = min(args.start_obj + args.max_objectives, len(objectives))
    selected_objectives = objectives[args.start_obj:end_obj]
    
    end_action = min(args.start_action + args.max_actions, len(actions))
    selected_actions = actions[args.start_action:end_action]
    
    print(f"\nExperiment Configuration:")
    print(f"  Objectives: {len(selected_objectives)} (indices {args.start_obj}-{end_obj-1})")
    print(f"  Actions: {len(selected_actions)} (indices {args.start_action}-{end_action-1})")
    print(f"  PAIR iterations: {args.iterations}")
    print(f"  Attacker model: {'Groq gpt-oss-20b' if args.use_groq_attacker else 'OpenAI GPT-4o-mini'}")
    print(f"  Mode: {args.mode}")
    
    # Run experiments
    all_results = {
        'experiment_config': {
            'max_objectives': args.max_objectives,
            'max_actions': args.max_actions,
            'start_obj': args.start_obj,
            'start_action': args.start_action,
            'iterations': args.iterations,
            'use_groq_attacker': args.use_groq_attacker,
            'mode': args.mode,
            'timestamp': datetime.now().isoformat()
        },
        'experiments': []
    }
    
    experiment_id = 1
    
    # Run experiments based on mode
    if args.mode in ['standard', 'both']:
        print(f"\n{'='*60}")
        print("RUNNING STANDARD MODE EXPERIMENTS")
        print(f"{'='*60}")
        
        for i, objective_data in enumerate(selected_objectives):
            result = run_pair_experiment(
                objective_data=objective_data,
                agentic_context="",  # Empty context for standard mode
                experiment_id=f"std_{experiment_id}",
                num_iterations=args.iterations,
                use_groq_attacker=args.use_groq_attacker
            )
            result['mode'] = 'standard'
            all_results['experiments'].append(result)
            experiment_id += 1
    
    if args.mode in ['agentic', 'both']:
        print(f"\n{'='*60}")
        print("RUNNING AGENTIC MODE EXPERIMENTS")
        print(f"{'='*60}")
        
        for i, objective_data in enumerate(selected_objectives):
            for j, action_data in enumerate(selected_actions):
                # Extract context from action data
                agentic_context = extract_agentic_context(action_data)
                
                result = run_pair_experiment(
                    objective_data=objective_data,
                    agentic_context=agentic_context,
                    experiment_id=f"agt_{experiment_id}",
                    num_iterations=args.iterations,
                    use_groq_attacker=args.use_groq_attacker
                )
                result['mode'] = 'agentic'
                result['action_label'] = action_data.get('label', 'unknown')
                result['action_risk'] = action_data.get('risk', 0.0)
                all_results['experiments'].append(result)
                experiment_id += 1
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"pair_agentic_results_{timestamp}.json"
    
    # Save results
    save_results(all_results, args.output)
    
    # Print summary
    total_experiments = len(all_results['experiments'])
    successful_experiments = sum(1 for exp in all_results['experiments'] 
                               if exp.get('success', False))
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful jailbreaks: {successful_experiments}")
    print(f"Success rate: {successful_experiments/total_experiments*100:.1f}%")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
