from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TrainingConfig:
    model_name_or_path: str = "openai/whisper-small"
    dataset_name: str = "ai4bharat/indicvoices_r"
    languages: Dict[str, str] = {
        "hi": "Hindi",
        "kn": "Kannada",
        "en": "English",
        "ta": "Tamil",
        "sa": "Sanskrit"
    }
    output_dir: str = "../../finetuned_multilingual"
    cache_dir: str = "./cache"
    num_train_epochs: int = 30
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    logging_steps: int = 100
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    fp16: bool = True
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    max_steps: int = -1  # -1 means train for num_train_epochs
    gradient_checkpointing: bool = True
    group_by_length: bool = True  # Reduces padding, speeds up training
    hub_model_id: str = None  # For pushing to Hugging Face Hub
    push_to_hub: bool = False
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
