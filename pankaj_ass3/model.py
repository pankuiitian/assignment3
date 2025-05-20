import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NetworkConfiguration
from model_components import EncoderNetwork, DecoderNetwork, AttentiveDecoderNetwork

class TransliterationEngine(nn.Module):
    """Full transliteration model combining encoder and decoder"""
    def __init__(self, config: NetworkConfiguration):
        super().__init__()
        self.config = config
        
        # Create encoder
        self.encoder = EncoderNetwork(config)
        
        # Create appropriate decoder based on config
        if config.apply_attention:
            self.decoder = AttentiveDecoderNetwork(config)
        else:
            self.decoder = DecoderNetwork(config)
            
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self, source, target=None, teacher_forcing_p=1.0, beam_size=1, return_attention_map=False):
        """
        Args:
            source: Source sequence [batch_size, seq_len]
            target: Target sequence for teacher forcing [batch_size, seq_len]
            teacher_forcing_p: Probability of using teacher forcing
            beam_size: Beam size for beam search decoding
            return_attention_map: Whether to return attention map
        Returns:
            loss: Loss value (if target is provided)
            accuracy: Accuracy value (if target is provided)
            decoder_outputs: Decoder outputs [batch_size, seq_len, vocab_size]
            attention_map: Attention map (if requested and available)
        """
        if return_attention_map and source.size(0) != 1:
            raise ValueError("Attention maps only supported for batch_size=1")
            
        # Encode source sequence
        enc_outputs, enc_hidden = self.encoder(source)
        
        # Prepare encoder hidden state for decoder
        dec_hidden = self._adapt_hidden_state(enc_hidden)
        
        # Decode
        outputs, attn_map = self.decoder(
            enc_outputs, dec_hidden, target, teacher_forcing_p, beam_size, return_attention_map)
        
        # Calculate loss and accuracy if target provided
        if target is None:
            return None, None, outputs, attn_map
            
        metrics = self._compute_metrics(outputs, target)
        return metrics[0], metrics[1], outputs, attn_map
    
    def _adapt_hidden_state(self, enc_hidden):
        encoder_type = self.config.encoder_name
        decoder_type = self.config.decoder_name
        is_bidir = self.config.encoder_bidirectional
        
        # Handle different RNN type combinations and bidirectionality
        if encoder_type == "LSTM" and decoder_type == "LSTM":
            h, c = enc_hidden
            if is_bidir:
                h = self._combine_directions(h)
                c = self._combine_directions(c)
            h = self._resize_layers(h)
            c = self._resize_layers(c)
            return (h, c)
            
        elif encoder_type == "LSTM" and decoder_type != "LSTM":
            h, _ = enc_hidden  # Discard cell state
            if is_bidir:
                h = self._combine_directions(h)
            return self._resize_layers(h)
            
        elif encoder_type != "LSTM" and decoder_type == "LSTM":
            if is_bidir:
                h = self._combine_directions(enc_hidden)
            else:
                h = enc_hidden
            h = self._resize_layers(h)
            c = torch.zeros_like(h)
            return (h, c)
            
        else:  # Both are GRU or RNN
            if is_bidir:
                h = self._combine_directions(enc_hidden)
            else:
                h = enc_hidden
            return self._resize_layers(h)
    
    def _combine_directions(self, hidden):
        num_layers = hidden.size(0) // 2
        batch_size = hidden.size(1)
        hidden_size = hidden.size(2)
        
        # Reshape to separate layers and directions
        hidden = hidden.view(num_layers, 2, batch_size, hidden_size)
        
        # Combine directions (sum)
        combined = torch.sum(hidden, dim=1)
        
        return combined
    
    def _resize_layers(self, hidden):
        src_layers = hidden.size(0)
        tgt_layers = self.config.decoder_num_layers
        
        if src_layers == tgt_layers:
            return hidden
        
        if src_layers < tgt_layers:
            # Add more layers by repeating the last layer
            padding = hidden[-1:].expand(tgt_layers - src_layers, -1, -1)
            return torch.cat([hidden, padding], dim=0)
        else:
            # Take only the needed layers
            return hidden[:tgt_layers]
    
    def _compute_metrics(self, logits, targets):
        batch_size, seq_len, vocab_size = logits.size()
        
        # Reshape for loss calculation
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        
        # Calculate loss
        loss = self.criterion(flat_logits, flat_targets)
        
        # Calculate accuracy (ignoring padding)
        mask = (flat_targets != 0)
        correct = (flat_logits.argmax(dim=1) == flat_targets) & mask
        total = mask.sum().item()
        
        if total > 0:
            accuracy = correct.sum().float() / total
        else:
            accuracy = torch.tensor(0.0).to(logits.device)
            
        return loss, accuracy