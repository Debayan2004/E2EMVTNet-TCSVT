import torch
import torch.nn as nn
from configs.config import MAX_TEXT_LENGTH, VOCAB_SIZE, MEMORY_DIM

TOKEN_DIM = MEMORY_DIM
MAX_WORD_LENGTH = MAX_TEXT_LENGTH
EMBED_DIM = MEMORY_DIM


class TokenFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM * 4),
            nn.GELU(),
            nn.Linear(MEMORY_DIM * 4, MEMORY_DIM)
        )

    def forward(self, x):
        return self.ffn(x)


class SharedTokenInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=MEMORY_DIM,
            num_heads=8,
            batch_first=True
        )
        self.ffn = TokenFFN()
        self.norm1 = nn.LayerNorm(MEMORY_DIM)
        self.norm2 = nn.LayerNorm(MEMORY_DIM)

    def forward(self, tokens):
        attn, _ = self.self_attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn)
        ffn = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn)
        global_token = tokens.mean(dim=1)

        return {
            "shared_tokens": tokens,
            "global_token": global_token
        }


class ProposalFeatureRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.LayerNorm(MEMORY_DIM),
            nn.GELU()
        )

    def forward(self, proposal_tokens):
        return self.refine(proposal_tokens)


class ROIProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(MEMORY_DIM),
            nn.GELU()
        )

    def forward(self, x):
        return self.project(x)


class FusionFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM * 4),
            nn.GELU(),
            nn.Linear(MEMORY_DIM * 4, MEMORY_DIM)
        )

    def forward(self, x):
        return self.ffn(x)


class CrossModalTokenFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_projection = ROIProjection()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=MEMORY_DIM,
            num_heads=8,
            batch_first=True
        )
        self.ffn = FusionFFN()
        self.norm1 = nn.LayerNorm(MEMORY_DIM)
        self.norm2 = nn.LayerNorm(MEMORY_DIM)

    def forward(self, shared_tokens, roi_tokens):
        roi = self.roi_projection(roi_tokens)

        fusion, _ = self.cross_attention(
            query=shared_tokens,
            key=roi,
            value=roi
        )

        fusion = self.norm1(shared_tokens + fusion)
        out = self.ffn(fusion)
        fusion = self.norm2(fusion + out)
        global_token = fusion.mean(dim=1)

        return {
            "fusion_tokens": fusion,
            "global_token": global_token,
            "roi_tokens": roi
        }


class CharacterEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.position = nn.Parameter(
            torch.randn(1, MAX_WORD_LENGTH, EMBED_DIM)
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        length = x.size(1)
        return x + self.position[:, :length]


class RecognitionContextProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_proj = nn.Sequential(
            nn.Linear(MEMORY_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU()
        )
        self.memory_global_proj = nn.Sequential(
            nn.Linear(MEMORY_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU()
        )

    def forward(self, context_output):
        tokens = context_output["fusion_tokens"]
        global_token = context_output["global_token"]

        tokens = self.memory_proj(tokens)
        global_token = self.memory_global_proj(global_token)
        global_token = global_token.unsqueeze(1)

        return tokens + global_token


class RecognitionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.character = CharacterEmbedding()
        self.context = RecognitionContextProjection()

    def forward(self, input_tokens, context_output):
        char_embedding = self.character(input_tokens)
        context = self.context(context_output)

        return {
            "character_embedding": char_embedding,
            "recognition_context": context
        }


class DecoderFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM * 4),
            nn.GELU(),
            nn.Linear(MEMORY_DIM * 4, MEMORY_DIM)
        )

    def forward(self, x):
        return self.ffn(x)


class MemoryGuidedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=MEMORY_DIM,
            num_heads=8,
            batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=MEMORY_DIM,
            num_heads=8,
            batch_first=True
        )
        self.ffn = DecoderFFN()
        self.norm1 = nn.LayerNorm(MEMORY_DIM)
        self.norm2 = nn.LayerNorm(MEMORY_DIM)
        self.norm3 = nn.LayerNorm(MEMORY_DIM)

        
    def forward(self, recognition_output):
        query = recognition_output["character_embedding"]
        memory = recognition_output["recognition_context"]

        length = query.size(1)
        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                device=query.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        self_attn, _ = self.self_attention(
            query,
            query,
            query,
            attn_mask=causal_mask
        )
        query = self.norm1(query + self_attn)

        cross_attn, _ = self.cross_attention(
            query,
            memory,
            memory
        )
        query = self.norm2(query + cross_attn)

        ffn = self.ffn(query)
        decoder_tokens = self.norm3(query + ffn)

        return {
            "decoder_tokens": decoder_tokens,
            "memory_context": memory
        }



class CharacterPredictionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(MEMORY_DIM)
        self.classifier = nn.Linear(MEMORY_DIM, VOCAB_SIZE)

    def forward(self, decoder_output):
        tokens = decoder_output["decoder_tokens"]
        tokens = self.norm(tokens)

        logits = self.classifier(tokens)
        probability = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probability, dim=-1)
        confidence = probability.max(dim=-1)[0]

        return {
            "recognition_tokens": tokens,
            "character_logits": logits,
            "character_probability": probability,
            "predicted_indices": prediction,
            "character_confidence": confidence
        }


class RecognitionOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_projection = nn.Sequential(
            nn.Linear(MEMORY_DIM, MEMORY_DIM),
            nn.GELU(),
            nn.LayerNorm(MEMORY_DIM)
        )

    def forward(self, decoder_output, prediction_output):
        decoder_tokens = decoder_output["decoder_tokens"]
        word_embedding = decoder_tokens.mean(dim=1)
        word_embedding = self.word_projection(word_embedding)

        return {
            "recognition_tokens": prediction_output["recognition_tokens"],
            "character_logits": prediction_output["character_logits"],
            "character_probability": prediction_output["character_probability"],
            "predicted_indices": prediction_output["predicted_indices"],
            "character_confidence": prediction_output["character_confidence"],
            "word_embedding": word_embedding
        }