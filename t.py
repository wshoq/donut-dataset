from axolotl.cli.train import do_cli
from axolotl.utils.config import validate_config

config_dict = {
    "base_model": "naver-clova-ix/donut-base",
    "datasets": [
        {
            "path": "./dataset/donut_dataset.jsonl",
            "type": "json",
            "field_input": "input",
            "field_output": "output",
            "field_image": "image_path"
        }
    ],
    "training_arguments": {
        "output_dir": "./outputs/donut-finetuned",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "learning_rate": 5e-5,
        "gradient_checkpointing": True,
        "remove_unused_columns": False
    }
}

# Walidacja dict → Pydantic
cfg_obj = validate_config(config_dict)

# Uruchom trening
do_cli(cfg_obj)
