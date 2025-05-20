import argparse
import wandb
from config import TrainingParameters, NetworkConfiguration
from trainer import ModelCoach

def setup_argument_parser():
    parser = argparse.ArgumentParser()
    
    group1 = parser.add_argument_group("Logging options")
    group1.add_argument("--wandb_entity", type=str, default="cs24m029-indian-institute-of-technology-madras")
    group1.add_argument("--wandb_project", type=str, default="dakshina-transliteration")
    group1.add_argument("--logging", action="store_true")
    
    group2 = parser.add_argument_group("Data options")
    group2.add_argument("--language", type=str, default="marathi")
    group2.add_argument("--data_path", type=str, default="C:\\Users\\pankaj\\Desktop\\dl\\DL_ass_3\\assignment3\\pankaj_ass3\\data")
    
    group3 = parser.add_argument_group("Training options")
    group3.add_argument("--batch_size", type=int, default=256)
    group3.add_argument("--num_workers", type=int, default=16)
    group3.add_argument("--learning_rate", type=float, default=0.003)
    group3.add_argument("--weight_decay", type=float, default=0.0005)
    group3.add_argument("--teacher_forcing_p", type=float, default=0.8)
    group3.add_argument("--max_epoch", type=int, default=10)
    group3.add_argument("--beam_size", type=int, default=3)
    
    group4 = parser.add_argument_group("Model options")
    group4.add_argument("--embedding_size", type=int, default=256)
    group4.add_argument("--hidden_size", type=int, default=256)
    group4.add_argument("--encoder_num_layers", type=int, default=1)
    group4.add_argument("--decoder_num_layers", type=int, default=1)
    group4.add_argument("--encoder_name", type=str, default="LSTM", choices=["RNN", "GRU", "LSTM"])
    group4.add_argument("--decoder_name", type=str, default="LSTM", choices=["RNN", "GRU", "LSTM"])
    group4.add_argument("--encoder_bidirectional", action="store_true", default=True)
    group4.add_argument("--dropout_p", type=float, default=0.3)
    group4.add_argument("--max_length", type=int, default=32)
    group4.add_argument("--apply_attention", action="store_true", default=False)
    
    return parser

def run_training():
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    enable_logging = args.logging or (args.wandb_entity and args.wandb_project)
    
    if enable_logging:
        model_type = f"{args.encoder_name}-{args.decoder_name}"
        run_name = f"{model_type}_beam{args.beam_size}_lr{args.learning_rate}_h{args.hidden_size}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )
    
    model_config = NetworkConfiguration(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_name=args.encoder_name,
        decoder_name=args.decoder_name,
        encoder_bidirectional=args.encoder_bidirectional,
        dropout_p=args.dropout_p,
        max_length=args.max_length,
        apply_attention=args.apply_attention
    )
    
    train_config = TrainingParameters(
        language=args.language,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        teacher_forcing_p=args.teacher_forcing_p,
        max_epoch=args.max_epoch,
        beam_size=args.beam_size,
        logging=enable_logging,
        model_config=model_config
    )
    
    try:
        coach = ModelCoach(train_config)
        coach.execute_training()
        coach.evaluate_model()
    finally:
        if enable_logging:
            wandb.finish()

if __name__ == "__main__":
    run_training()
