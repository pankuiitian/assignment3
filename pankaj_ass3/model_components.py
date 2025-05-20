import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from config import NetworkConfiguration

class EncoderNetwork(nn.Module):
    def __init__(self, config: NetworkConfiguration):
        super().__init__()
        self.embedding = nn.Embedding(config.source_vocab_size, config.embedding_size)
        self.dropout_in = nn.Dropout(config.dropout_p)
        self.dropout_out = nn.Dropout(config.dropout_p)
        
        rnn_cls = getattr(nn, config.encoder_name)
        directions = 2 if config.encoder_bidirectional else 1
        
        self.rnn = rnn_cls(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size // directions if config.encoder_bidirectional else config.hidden_size,
            num_layers=config.encoder_num_layers,
            batch_first=True,
            dropout=config.dropout_p if config.encoder_num_layers > 1 else 0,
            bidirectional=config.encoder_bidirectional
        )

    def forward(self, x):
        x_emb = self.dropout_in(self.embedding(x))
        outputs, hidden = self.rnn(x_emb)
        outputs = self.dropout_out(outputs)
        return outputs, hidden


class AttentionMechanism(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size, src_len, _ = encoder_outputs.shape
        
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden_expanded, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class BaseDecoder(nn.Module):
    def __init__(self, config: NetworkConfiguration):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.target_vocab_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout_p)
        
    def init_hidden(self, hidden):
        if self.config.decoder_name == "LSTM" and not isinstance(hidden, tuple):
            zeros = torch.zeros_like(hidden)
            return (hidden, zeros)
        return hidden


class DecoderNetwork(BaseDecoder):
    def __init__(self, config: NetworkConfiguration):
        super().__init__(config)
        
        rnn_cls = getattr(nn, config.decoder_name)
        self.rnn = rnn_cls(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.decoder_num_layers,
            batch_first=True,
            dropout=config.dropout_p if config.decoder_num_layers > 1 else 0
        )
        
        self.out = nn.Linear(config.hidden_size, config.target_vocab_size)

    def forward(self, encoder_outputs, encoder_hidden, target=None, teacher_forcing_p=1.0, 
                beam_size=1, return_attention_map=False):
        batch_size = encoder_outputs.size(0)
        max_len = target.shape[1] if target is not None else self.config.max_length
        device = encoder_outputs.device
        
        hidden = self.init_hidden(encoder_hidden)
        input_token = torch.full((batch_size, 1), self.config.decoder_SOS, dtype=torch.long, device=device)
        
        outputs = []
        
        for t in range(max_len):
            output, hidden = self._step(input_token, hidden)
            outputs.append(output)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_p
            if teacher_force and target is not None:
                input_token = target[:, t:t+1]
            else:
                input_token = output.argmax(2)
        
        return torch.cat(outputs, dim=1), None
        
    def _step(self, input_token, hidden):
        embedded = self.dropout(self.embedding(input_token))
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output)
        return output, hidden


class AttentiveDecoderNetwork(BaseDecoder):
    def __init__(self, config: NetworkConfiguration):
        super().__init__(config)
        
        enc_dim = config.hidden_size * 2 if config.encoder_bidirectional else config.hidden_size
        self.attention = AttentionMechanism(enc_dim, config.hidden_size)
        
        rnn_cls = getattr(nn, config.decoder_name)
        self.rnn = rnn_cls(
            input_size=config.embedding_size + enc_dim,
            hidden_size=config.hidden_size,
            num_layers=config.decoder_num_layers,
            batch_first=True,
            dropout=config.dropout_p if config.decoder_num_layers > 1 else 0
        )
        
        self.out = nn.Linear(config.hidden_size, config.target_vocab_size)
        self.BeamNode = namedtuple("BeamNode", ["tokens", "score", "state"])

    def forward(self, encoder_outputs, encoder_hidden, target=None, teacher_forcing_p=1.0, 
                beam_size=1, return_attention_map=False):
        if beam_size > 1:
            return self.beam_search(encoder_outputs, encoder_hidden, beam_size)
        return self.greedy_decode(encoder_outputs, encoder_hidden, target, teacher_forcing_p, return_attention_map)
        
    def greedy_decode(self, encoder_outputs, encoder_hidden, target, teacher_forcing_p, return_attention_map):
        batch_size = encoder_outputs.size(0)
        max_len = target.shape[1] if target is not None else self.config.max_length
        device = encoder_outputs.device
        
        hidden = self.init_hidden(encoder_hidden)
        input_token = torch.full((batch_size, 1), self.config.decoder_SOS, dtype=torch.long, device=device)
        
        outputs = []
        attention_weights = []
        
        for t in range(max_len):
            output, hidden, attn_wt = self._step(input_token, hidden, encoder_outputs, return_attention_map)
            outputs.append(output)
            
            if return_attention_map:
                attention_weights.append(attn_wt)
                
            teacher_force = torch.rand(1).item() < teacher_forcing_p
            if teacher_force and target is not None:
                input_token = target[:, t:t+1]
            else:
                input_token = output.argmax(2)
        
        decoder_outputs = torch.cat(outputs, dim=1)
        attn_map = torch.stack(attention_weights, dim=1) if attention_weights else None
        
        return decoder_outputs, attn_map
        
    def _step(self, input_token, hidden, encoder_outputs, return_attn_weights=False):
        embedded = self.dropout(self.embedding(input_token))
        
        if isinstance(hidden, tuple):
            query = hidden[0][-1]
        else:
            query = hidden[-1]
            
        attn_weights = self.attention(query, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        output = self.out(output)
        return output, hidden, attn_weights if return_attn_weights else None

    def beam_search(self, encoder_outputs, encoder_hidden, beam_size):
        batch_size = encoder_outputs.size(0)
        max_len = self.config.max_length
        device = encoder_outputs.device
        
        results = []
        
        for b in range(batch_size):
            enc_out = encoder_outputs[b:b+1]
            
            if isinstance(encoder_hidden, tuple):
                h, c = encoder_hidden
                hidden = (h[:, b:b+1].contiguous(), c[:, b:b+1].contiguous())
            else:
                hidden = encoder_hidden[:, b:b+1].contiguous()
            
            beam = [(
                [self.config.decoder_SOS], 
                0.0,
                hidden
            )]
            
            for _ in range(max_len - 1):
                new_beam = []
                
                for tokens, score, h in beam:
                    if tokens[-1] == 0:
                        new_beam.append((tokens, score, h))
                        continue
                    
                    input_t = torch.tensor([[tokens[-1]]], device=device)
                    out, new_h, _ = self._step(input_t, h, enc_out)
                    
                    probs = F.log_softmax(out.squeeze(0), dim=-1)
                    
                    topk_probs, topk_idx = probs.topk(beam_size)
                    
                    for i in range(beam_size):
                        new_tokens = tokens + [topk_idx[0, i].item()]
                        new_score = score + topk_probs[0, i].item()
                        new_beam.append((new_tokens, new_score, new_h))
                
                beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
            
            best_seq, _, _ = max(beam, key=lambda x: x[1])
            results.append(best_seq)
        
        padded_results = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(results):
            length = min(len(seq), max_len)
            padded_results[i, :length] = torch.tensor(seq[:length], device=device)
        
        return padded_results, None