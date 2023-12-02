SCHEMAS = {
    "TRAIN_XL_LORA": {
        "fn":{
            "type": str,
            "required": True,
        },
        "dataset_repo": {
            "type": str,
            'required': True,
        },
        "model_name":{
            "type": str,
            'required': True,
        },
        "max_train_epochs":{
            "type": int,
            'required': True,
        },
        "train_batch_size":{
            "type": int,
            'required': True,
        },
        "save_every_n_epochs":{
            "type": int,
            'required': False,
            "default": 10,
        },
        "learning_rate":{
            "type": float,
            'required': False,
            "default": 1e-4,
        },
        "network_dim":{
            "type": int,
            'required': False,
            "default": 32,
        },
        "network_alpha":{
            "type": int,
            "required": False,
            "default": 32,
        }

    }
}

INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'DDIM'
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 25
    },
    'refiner_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'image_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    }
}
