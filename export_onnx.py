# export_onnx.py
import os

import torch
import torch.nn as nn


class SurgicalFeatureNet(nn.Module):
    """초경량 수술 비전 백본 (128x128 입력, 4개 클래스/마진 판정 프록시)"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main():
    model = SurgicalFeatureNet()
    model.eval()

    # Model Repository 구조에 맞춘 저장 경로
    save_dir = "model_repository/surgical_feature_net/1"
    os.makedirs(save_dir, exist_ok=True)
    onnx_path = os.path.join(save_dir, "model.onnx")

    # Dynamic Batching을 위한 가변 배치 더미 텐서
    dummy_input = torch.randn(1, 3, 128, 128, requires_grad=False)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_frame"],
        output_names=["logits"],
        dynamic_axes={
            "input_frame": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"[SUCCESS] ONNX Model exported to: {onnx_path}")


if __name__ == "__main__":
    main()
