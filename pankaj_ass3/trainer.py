import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from config import TrainingParameters, NetworkConfiguration
from model import TransliterationEngine
from dataset import TransliterationDataset, batch_collate, prepare_vocabularies

class ModelCoach:
    def __init__(self, config: TrainingParameters):
        self.config = config
        self.logging = config.logging
        
        prepare_vocabularies(config.train_data_path, config.dev_data_path)
        
        self.datasets = {}
        self.dataloaders = {}
        
        for split, path, normalize in [
            ('train', config.train_data_path, True),
            ('valid', config.dev_data_path, False),
            ('test', config.test_data_path, False)
        ]:
            self.datasets[split] = TransliterationDataset(path, normalize=normalize)
            self.dataloaders[split] = DataLoader(
                self.datasets[split],
                batch_size=config.batch_size,
                collate_fn=batch_collate,
                shuffle=(split == 'train'),
                num_workers=config.num_workers,
                persistent_workers=True,
                pin_memory=True
            )
        
        model_config_dict = vars(config.model_config).copy()
        model_config_dict.update({
            'decoder_SOS': self.datasets['train'].target.SOS,
            'source_vocab_size': self.datasets['train'].source.vocab_size,
            'target_vocab_size': self.datasets['train'].target.vocab_size,
        })
        
        model_config = NetworkConfiguration(**model_config_dict)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TransliterationEngine(model_config).to(self.device)
        
        self.precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        print(f"Using {self.device} with {self.precision} precision")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    def execute_training(self):
        print(f"Starting training for {self.config.max_epoch} epochs")
        
        history = []
        
        for epoch in range(1, self.config.max_epoch + 1):
            train_stats = self._process_epoch(epoch, 'train')
            valid_stats = self._process_epoch(epoch, 'valid')
            
            epoch_stats = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'valid_{k}': v for k, v in valid_stats.items()},
            }
            
            history.append(epoch_stats)
            
            log_str = f"Epoch {epoch}/{self.config.max_epoch} | "
            log_str += f"Train loss: {epoch_stats['train_loss']:.4f} | "
            log_str += f"Train acc: {epoch_stats['train_acc']:.4f} | "
            log_str += f"Valid loss: {epoch_stats['valid_loss']:.4f} | "
            log_str += f"Valid acc: {epoch_stats['valid_acc']:.4f}"
            print(log_str)
            
            if self.logging:
                wandb.log(epoch_stats)
        
        print("Training complete!")
        return history

    def _process_epoch(self, epoch, split):
        is_train = split == 'train'
        self.model.train(is_train)
        torch.set_grad_enabled(is_train)
        
        total_loss = 0
        total_acc = 0
        total_batches = 0
        
        desc = f"Epoch {epoch}/{self.config.max_epoch} [{split.capitalize()}]"
        
        for source, target in tqdm(self.dataloaders[split], desc=desc):
            source = source.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            with torch.autocast(device_type=self.device.type, dtype=self.precision):
                loss, acc, *_ = self.model(
                    source, target, self.config.teacher_forcing_p)
            
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item() 
            total_batches += 1
        
        return {
            'loss': total_loss / total_batches,
            'acc': total_acc / total_batches
        }

    def evaluate_model(self, plot_attention=False):
        print("Running inference on test set")
        self.model.eval()
        
        all_refs = []
        all_hyps = []
        
        with torch.no_grad():
            for source, target in tqdm(self.dataloaders['test'], desc="Testing"):
                source = source.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                with torch.autocast(device_type=self.device.type, dtype=self.precision):
                    _, _, outputs, _ = self.model(source, beam_size=self.config.beam_size)
                
                for i in range(source.size(0)):
                    ref_ids = [t.item() for t in target[i] if t.item() not in [0]]
                    hyp_ids = [t.item() for t in outputs[i] if t.item() not in [0]]
                    
                    all_refs.append(ref_ids)
                    all_hyps.append(hyp_ids)
        
        bleu_score, targets, preds = self._compute_bleu(all_refs, all_hyps)
        print(f"Test BLEU Score: {bleu_score:.4f}")
        
        if self.logging:
            wandb.log({"test_bleu": bleu_score})
            
            samples = min(10, len(targets))
            examples = [[targets[i], preds[i]] for i in range(samples)]
            table = wandb.Table(columns=["Target", "Prediction"], data=examples)
            wandb.log({"predictions": table})
        
        return bleu_score, targets, preds

    def _compute_bleu(self, refs, hyps):
        bleu_func = SmoothingFunction().method1
        scores = []
        targets = []
        predictions = []
        
        for ref_ids, hyp_ids in zip(refs, hyps):
            if not hyp_ids:
                continue
                
            ref_str = self.datasets['test'].target.decode(ref_ids)
            hyp_str = self.datasets['test'].target.decode(hyp_ids)
            
            targets.append(ref_str)
            predictions.append(hyp_str)
            
            score = sentence_bleu([ref_ids], hyp_ids, smoothing_function=bleu_func)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score, targets, predictions

    def predict_samples(self, source_words, visualize_attention=False):
        self.model.eval()
        results = []
        attention_data = []
        
        with torch.no_grad():
            for word in source_words:
                source_ids = self.datasets['train'].source.encode(word)
                source_tensor = torch.tensor([source_ids], device=self.device)
                
                with torch.autocast(device_type=self.device.type, dtype=self.precision):
                    _, _, outputs, attn = self.model(
                        source_tensor, 
                        beam_size=self.config.beam_size,
                        return_attention_map=visualize_attention
                    )
                
                pred_ids = [t.item() for t in outputs[0] if t.item() != 0]
                pred_word = self.datasets['train'].target.decode(pred_ids)
                
                results.append((word, pred_word))
                
                if visualize_attention and attn is not None:
                    attention_data.append({
                        'source_word': word,
                        'target_word': pred_word,
                        'attention_map': attn.squeeze().cpu().numpy()
                    })
        
        for src, pred in results:
            print(f"Source: {src} → Prediction: {pred}")
        
        if visualize_attention and attention_data:
            while len(attention_data) < 9:
                extend_by = min(len(attention_data), 9 - len(attention_data))
                attention_data.extend(attention_data[:extend_by])
            
            self._plot_attention(attention_data[:9])
        
        return results

    def _plot_attention(self, data, filename="attention_maps.png"):
        fig = plt.figure(figsize=(16, 16))
        
        for i, item in enumerate(data):
            ax = plt.subplot(3, 3, i + 1)
            attention = item['attention_map']
            src = item['source_word']
            tgt = item['target_word']
            
            im = ax.imshow(attention, cmap='viridis')
            
            ax.set_xlabel(f'Source: {src}')
            ax.set_ylabel(f'Target: {tgt}')
            
            ax.set_xticks(range(len(src)))
            ax.set_xticklabels(list(src))
            ax.set_yticks(range(len(tgt)))
            ax.set_yticklabels(list(tgt))
            
            ax.set_title(f"{src} → {tgt}")
        
        for j in range(len(data), 9):
            ax = plt.subplot(3, 3, j + 1)
            ax.axis('off')

        plt.tight_layout()
        cbar = fig.colorbar(im, ax=fig.axes, orientation='vertical', fraction=0.015, pad=0.04)
        
        plt.savefig(filename)
        plt.close()
        
        if self.logging:
            wandb.log({"attention_visualization": wandb.Image(filename)})