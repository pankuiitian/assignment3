from dataclasses import dataclass, field
import os

LANG_MAP = {}
for k, v in [("bengali", "bn"), ("gujarati", "gu"), ("hindi", "hi"), ("kannada", "kn"),
             ("malayalam", "ml"), ("marathi", "mr"), ("punjabi", "pa"), ("sindhi", "sd"),
             ("sinhala", "si"), ("tamil", "ta"), ("telugu", "te"), ("urdu", "ur")]:
    LANG_MAP[k] = v

@dataclass
class NetworkConfiguration:
    source_vocab_size: int = 500
    target_vocab_size: int = 500
    embedding_size: int = 256
    hidden_size: int = 256
    encoder_num_layers: int = 3
    decoder_num_layers: int = 2
    encoder_name: str = "GRU"
    decoder_name: str = "GRU"
    encoder_bidirectional: bool = True
    decoder_bidirectional: bool = False
    dropout_p: float = 0.3
    max_length: int = 32
    decoder_SOS: int = 0
    apply_attention: bool = True
    
    def __post_init__(self):
        valid_types = set(["RNN", "GRU", "LSTM"])
        if self.encoder_name not in valid_types:
            raise ValueError(f"Encoder must be one of: {', '.join(valid_types)}")
        if self.decoder_name not in valid_types:
            raise ValueError(f"Decoder must be one of: {', '.join(valid_types)}")

@dataclass
class TrainingParameters:
    language: str = "marathi"
    data_path: str = "C:\\Users\\pankaj\\Desktop\\dl\\DL_ass_3\\assignment3\\pankaj_ass3\\data"
    batch_size: int = 256
    num_workers: int = 16
    learning_rate: float = 0.003
    weight_decay: float = 0.0005
    teacher_forcing_p: float = 0.8
    max_epoch: int = 10
    beam_size: int = 1
    logging: bool = False
    model_config: NetworkConfiguration = field(default_factory=NetworkConfiguration)
    
    def __post_init__(self):
        if self.language not in LANG_MAP:
            raise ValueError(f"Language '{self.language}' not supported. Options: {', '.join(LANG_MAP.keys())}")
        
        code = LANG_MAP[self.language]
        template = os.path.join(self.data_path, f"{code}/lexicons/{code}.translit.sampled.{{}}.tsv")
        self.train_data_path = template.format("train")
        self.dev_data_path = template.format("dev")
        self.test_data_path = template.format("test")
        
        for path in [self.train_data_path, self.dev_data_path, self.test_data_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")