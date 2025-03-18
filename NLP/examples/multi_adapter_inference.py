"""
Multi-Adapter Inference Example

This script demonstrates how to use the InferencePipeLine class with multiple adapters,
dynamically loading and switching between them at runtime.
"""

import os
import sys
import argparse
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.pipelines import InferencePipeLine

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Adapter Inference Example")
    
    # Required arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["text_generation", "summarization", "translation", "question_answering"],
                        help="The type of task to perform")
    
    # Adapter directories
    parser.add_argument("--adapters_dir", type=str, default="./adapters",
                        help="Directory containing adapter folders")
    parser.add_argument("--cache_dir", type=str, default="./adapter_cache",
                        help="Directory to cache downloaded adapters")
    
    # Hugging Face adapters
    parser.add_argument("--hf_adapters", type=str, nargs="*", default=[],
                        help="List of Hugging Face adapter repo IDs to use")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token for private repositories")
    
    # Inference options
    parser.add_argument("--max_adapters", type=int, default=5,
                        help="Maximum number of adapters to keep in memory")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    
    # New option for automatic adapter selection
    parser.add_argument("--auto_adapter", action="store_true",
                        help="Let the system decide which adapter to use")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up adapter configuration
    adapter_config = {
        'cache_dir': args.cache_dir,
        'max_adapters': args.max_adapters,
        'preload_adapters': []
    }
    
    # Add local adapters to preload list if adapters_dir exists
    if os.path.exists(args.adapters_dir):
        for adapter_folder in os.listdir(args.adapters_dir):
            adapter_path = os.path.join(args.adapters_dir, adapter_folder)
            if os.path.isdir(adapter_path):
                adapter_config['preload_adapters'].append({
                    'source': 'local',
                    'adapter_id': adapter_folder,
                    'adapter_name': adapter_folder,
                    'path': adapter_path
                })
    
    # Add Hugging Face adapters to preload list
    for hf_adapter in args.hf_adapters:
        adapter_id = hf_adapter.replace('/', '_')
        adapter_config['preload_adapters'].append({
            'source': 'huggingface',
            'adapter_id': adapter_id,
            'repo_id': hf_adapter,
            'use_auth_token': args.hf_token
        })
    
    # Initialize the inference pipeline with adapter support
    print(f"Initializing inference pipeline with model: {args.model_name_or_path}")
    pipeline = InferencePipeLine(
        task_type=args.task_type,
        checkpoint=args.model_name_or_path,
        adapter_config=adapter_config
    )
    
    # Show available adapters
    adapters = pipeline.list_adapters()
    if adapters:
        print(f"Loaded {len(adapters)} adapters:")
        for adapter in adapters:
            print(f"  - {adapter['id']} ({adapter['name']})")
    else:
        print("No adapters loaded")
    
    # Interactive mode
    print("\n=== Interactive Inference ===")
    print("Type 'exit' to quit, 'adapters' to list adapters, 'adapter NAME' to switch adapter")
    print("Additional commands: 'register_local PATH ID', 'register_hf REPO_ID ID', 'unregister ID'")
    
    current_adapter = None
    while True:
        # Display prompt with current adapter
        if current_adapter:
            prompt = f"[{current_adapter}]> "
        else:
            prompt = "[base]> "
        
        # Get user input
        user_input = input(prompt)
        
        # Handle commands
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'adapters':
            adapters = pipeline.list_adapters()
            print(f"Available adapters ({len(adapters)}):")
            for adapter in adapters:
                status = "LOADED" if adapter["is_loaded"] else "not loaded"
                print(f"  - {adapter['id']} ({adapter['name']}) [{status}]")
            continue
        elif user_input.lower().startswith('adapter '):
            adapter_name = user_input[8:].strip()
            if adapter_name.lower() == 'none' or adapter_name.lower() == 'base':
                current_adapter = None
                print("Switched to base model")
            else:
                # Check if adapter exists
                adapters = pipeline.list_adapters()
                adapter_ids = [a['id'] for a in adapters]
                if adapter_name in adapter_ids:
                    current_adapter = adapter_name
                    print(f"Switched to adapter: {current_adapter}")
                else:
                    print(f"Adapter '{adapter_name}' not found")
            continue
        elif user_input.lower().startswith('register_local '):
            # Format: register_local PATH ID [NAME]
            args = user_input[14:].strip().split()
            if len(args) < 2:
                print("Usage: register_local PATH ID [NAME]")
                continue
                
            path = args[0]
            adapter_id = args[1]
            adapter_name = args[2] if len(args) > 2 else adapter_id
            
            try:
                pipeline.adapter_manager.register_adapter_from_local(
                    adapter_id=adapter_id,
                    adapter_path=path,
                    adapter_name=adapter_name
                )
                print(f"Registered local adapter: {adapter_id}")
            except Exception as e:
                print(f"Failed to register adapter: {str(e)}")
            continue
        elif user_input.lower().startswith('register_hf '):
            # Format: register_hf REPO_ID ID [NAME]
            args = user_input[12:].strip().split()
            if len(args) < 2:
                print("Usage: register_hf REPO_ID ID [NAME]")
                continue
                
            repo_id = args[0]
            adapter_id = args[1]
            adapter_name = args[2] if len(args) > 2 else adapter_id
            
            try:
                pipeline.adapter_manager.register_adapter_from_huggingface(
                    adapter_id=adapter_id,
                    repo_id=repo_id,
                    adapter_name=adapter_name,
                    use_auth_token=args.hf_token
                )
                print(f"Registered Hugging Face adapter: {adapter_id}")
            except Exception as e:
                print(f"Failed to register adapter: {str(e)}")
            continue
        elif user_input.lower().startswith('unregister '):
            # Format: unregister ID
            adapter_id = user_input[11:].strip()
            
            try:
                result = pipeline.adapter_manager.unregister_adapter(adapter_id)
                if result:
                    print(f"Unregistered adapter: {adapter_id}")
                    if current_adapter == adapter_id:
                        current_adapter = None
                        print("Switched to base model")
                else:
                    print(f"Adapter not found: {adapter_id}")
            except Exception as e:
                print(f"Failed to unregister adapter: {str(e)}")
            continue
        
        # Regular inference
        if not user_input:
            continue
        
        # Measure inference time
        start_time = time.time()
        
        # Run inference
        try:
            result = pipeline.run(
                user_input, 
                adapter_id=current_adapter, 
                max_length=args.max_length,
                do_sample=args.do_sample,
                temperature=args.temperature
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            print(f"\nResult [{inference_time:.2f}s]:")
            print(result)
            print()
        except Exception as e:
            print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main() 