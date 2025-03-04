from torch import nn

from multilingual_speech_emotion_recognition.models.fusion_model import \
    FusionModel
from multilingual_speech_emotion_recognition.models.handcrafted_feature_extractor import \
    HandcraftedAcousticEncoder
from multilingual_speech_emotion_recognition.models.pretrained_speech_encoder import \
    PretrainedSpeechEncoder


class KuralNet(nn.Module):
    def __init__(self, handcraft_dim=40, pretrained_dim=512, num_emotions=7):
        super(KuralNet, self).__init__()
        self.handcraft_net = HandcraftedAcousticEncoder(
            input_shape=handcraft_dim, num_classes=num_emotions
        )
        self.whisper_net = PretrainedSpeechEncoder(
            input_shape=pretrained_dim, num_classes=num_emotions
        )
        self.fusion_net = FusionModel(input_dim=128, num_emotions=num_emotions)

        self.train_handcraft = True
        self.train_whisper = True
        self.train_fusion = True

    def set_training_mode(
        self, train_handcraft=True, train_whisper=True, train_fusion=True
    ):
        self.train_handcraft = train_handcraft
        self.train_whisper = train_whisper
        self.train_fusion = train_fusion

        for param in self.handcraft_net.parameters():
            param.requires_grad = train_handcraft
        for param in self.whisper_net.parameters():
            param.requires_grad = train_whisper
        for param in self.fusion_net.parameters():
            param.requires_grad = train_fusion

    def forward(self, handcraft_features=None, whisper_features=None):
        handcraft_out = None
        whisper_out = None

        if handcraft_features is not None:
            handcraft_out = self.handcraft_net(handcraft_features)
        if whisper_features is not None:
            whisper_out = self.whisper_net(whisper_features)

        output = self.fusion_net(handcraft_out, whisper_out)
        return output
