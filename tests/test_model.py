import torch
from kuralnet.model.attention import TimeStepAttentionPooling, FeatureWiseAttentionPooling


def test_attention_time_step():
    batch_size, time_steps, embedding_dim = 1, 1500, 768
    audio_embeddings = torch.randn(batch_size, time_steps, embedding_dim)  # Simulated embeddings

    attention_pooling = TimeStepAttentionPooling(embedding_dim)
    pooled_embedding = attention_pooling(audio_embeddings)

    assert pooled_embedding.shape[0] == 1
    assert pooled_embedding.shape[1] == embedding_dim

def test_attention_feature_wise():
    batch_size, time_steps, embedding_dim = 1, 1500, 768
    audio_embeddings = torch.randn(batch_size, time_steps, embedding_dim)  # Simulated embeddings

    attention_pooling = FeatureWiseAttentionPooling(time_steps)
    pooled_embedding = attention_pooling(audio_embeddings)

    assert pooled_embedding.shape[0] == 1
    assert pooled_embedding.shape[1] == embedding_dim