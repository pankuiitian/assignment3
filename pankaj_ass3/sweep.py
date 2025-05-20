import wandb
from config import TrainingParameters, NetworkConfiguration
from trainer import ModelCoach
import os

DATA_PATH = "C:\\Users\\pankaj\\Desktop\\dl\\DL_ass_3\\assignment3\\pankaj_ass3\\data"
LANGUAGE = "marathi"
ENTITY = "cs24m029-indian-institute-of-technology-madras"
PROJECT = "dakshina-transliteration"

def configure_hyperparameter_sweep():
    return {
        "method": "bayes",
        "project": PROJECT,
        "entity": ENTITY,
        "name": f"{LANGUAGE}-sweep-01_withoutAttention",
        "metric": {"name": "valid_acc", "goal": "maximize"},
        "parameters": {
            "embedding_size": {"values": [256, 128]},
            "encoder_num_layers": {"values": [2, 3]},
            "decoder_num_layers": {"values": [1, 2]},
            "hidden_size": {"values": [128, 256]},
            "encoder_name": {"values": ["RNN", "GRU", "LSTM"]},
            "decoder_name": {"values": ["RNN", "GRU", "LSTM"]},
            "dropout_p": {"values": [0.2, 0.3]},
            "learning_rate": {"values": [0.001, 0.003]},
            "teacher_forcing_p": {"values": [0.5, 0.8]},
            "apply_attention": {"values": [False]},
            "encoder_bidirectional": {"values": [True]},
            "beam_size": {"values": [1]}
        }
    }

def train_sweep_model():
    run = wandb.init()
    
    params = wandb.config
    name_parts = [
        f"{params.encoder_name}-{params.decoder_name}",
        f"beam{params.beam_size}", 
        f"lr{params.learning_rate}",
        f"h{params.hidden_size}",
        f"e{params.embedding_size}",
        f"dr{params.dropout_p}"
    ]
    run.name = "_".join(name_parts)
    
    model_cfg = NetworkConfiguration(
        embedding_size=params.embedding_size,
        hidden_size=params.hidden_size,
        encoder_num_layers=params.encoder_num_layers,
        decoder_num_layers=params.decoder_num_layers,
        encoder_name=params.encoder_name,
        decoder_name=params.decoder_name,
        encoder_bidirectional=params.encoder_bidirectional,
        dropout_p=params.dropout_p,
        apply_attention=params.apply_attention,
    )
    
    train_cfg = TrainingParameters(
        language=LANGUAGE,
        data_path=DATA_PATH,
        learning_rate=params.learning_rate,
        teacher_forcing_p=params.teacher_forcing_p,
        beam_size=params.beam_size,
        batch_size=256,
        num_workers=os.cpu_count() or 16,
        weight_decay=0.0005,
        logging=True,
        model_config=model_cfg
    )
    
    coach = ModelCoach(train_cfg)
    coach.execute_training()
    wandb.finish()

if __name__ == "__main__":
    sweep_config = configure_hyperparameter_sweep()
    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT)
    print(f"Created sweep: {sweep_id}")
    wandb.agent(sweep_id, function=train_sweep_model, count=50)
