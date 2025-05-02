import numpy as np
import torch.nn as nn

from kuralnet.models.classification_head import ClassificationHead
from kuralnet.models.fusion import AttentionFusion
from kuralnet.models.pretrained_feature_extractor import (
    WhisperFeatureExtractor,
)
from kuralnet.models.pretrained_processor import WhisperLayersProcessor
from kuralnet.models.traditional_feature_extractor import (
    TraditionalFeatureExtractor,
)
from kuralnet.models.tradtional_processor import TraditionalLayersProcessor

FEATURE_DIM = 128
FUSION_DIM = 128
NUM_HEADS = 4
NUM_CLASSES = 5


class KuralNet(nn.Module):
    """
    Full model combining Whisper and traditional features with attention fusion.
    """

    def __init__(
        self,
        feature_dim=FEATURE_DIM,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
    ):

        super(KuralNet, self).__init__()
        self.whisper_feature_extractor = WhisperFeatureExtractor()
        self.whisper_processor = WhisperLayersProcessor(output_dim=feature_dim)
        self.trad_processor = TraditionalLayersProcessor(output_dim=feature_dim)
        self.fusion = AttentionFusion(
            feature_dim=feature_dim, fusion_dim=fusion_dim, num_heads=num_heads
        )
        self.classifier = ClassificationHead(
            in_dim=fusion_dim, num_classes=num_classes
        )

    def forward(self, whisper_feats,trad_feats):

        out_w = self.whisper_processor(whisper_feats)  # (batch, feature_dim)
        out_t = self.trad_processor(trad_feats)  # (batch, feature_dim)
        # Fuse via attention
        fused = self.fusion(out_w, out_t)  # (batch, fusion_dim)
        # Classify
        output = self.classifier(fused)  # (batch, num_classes)
        return output
