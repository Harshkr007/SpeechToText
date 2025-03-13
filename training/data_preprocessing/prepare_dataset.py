import datasets
from datasets import load_dataset, Audio, concatenate_datasets
import re
from tqdm import tqdm

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\"\%\'\"\�\']'
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def prepare_dataset(config):
    all_train_datasets = []
    all_eval_datasets = []
    
    # Track total examples
    total_train = 0
    total_eval = 0
    
    try:
        for lang_code in tqdm(config.languages.keys(), desc="Processing languages"):
            print(f"\nLoading dataset for {config.languages[lang_code]}...")
            
            try:
                # Load dataset for specific language
                dataset = load_dataset(
                    config.dataset_name,
                    lang_code,
                    cache_dir=config.cache_dir
                )
                
                # Add language label and processing
                dataset = dataset.map(
                    lambda x: {
                        "language": lang_code,
                        "text": x["text"].lower().strip()
                    },
                    desc=f"Processing {lang_code} text"
                )
                
                # Split if needed
                if "validation" not in dataset:
                    dataset = dataset["train"].train_test_split(
                        test_size=0.1, seed=42
                    )
                    train_data = dataset["train"]
                    eval_data = dataset["test"]
                else:
                    train_data = dataset["train"]
                    eval_data = dataset["validation"]
                
                # Update counts
                total_train += len(train_data)
                total_eval += len(eval_data)
                
                all_train_datasets.append(train_data)
                all_eval_datasets.append(eval_data)
                
                print(f"Added {len(train_data)} training and {len(eval_data)} validation examples for {lang_code}")
                
            except Exception as e:
                print(f"Error processing {lang_code}: {str(e)}")
                continue
        
        print(f"\nTotal dataset size:")
        print(f"Training: {total_train} examples")
        print(f"Validation: {total_eval} examples")
        
        # Combine datasets
        combined_train = concatenate_datasets(all_train_datasets)
        combined_eval = concatenate_datasets(all_eval_datasets)
        
        # Create final dataset
        final_dataset = datasets.DatasetDict({
            "train": combined_train,
            "validation": combined_eval
        })
        
        # Resample audio
        final_dataset = final_dataset.cast_column(
            "audio",
            Audio(sampling_rate=16000)
        )
        
        return final_dataset
        
    except Exception as e:
        raise Exception(f"Error preparing dataset: {str(e)}")
