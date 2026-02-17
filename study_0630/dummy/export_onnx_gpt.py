import torch
from vltseg import VLTSeg  # vltseg PyTorch 모델 클래스 임포트

# 1. 모델 로드 및 eval 모드 설정
model = VLTSeg()
model.load_state_dict(torch.load("vltseg_checkpoint.pth"))  # 가중치 파일 경로
model.eval()

# 2. 더미 입력 생성 (배치 1, RGB 3채널, 크기 512x512 등 모델 입력 크기에 맞게)
dummy_input = torch.randn(1, 3, 1022, 1022)

# 3. ONNX export
torch.onnx.export(
    model,
    dummy_input,
    "vltseg.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    verbose=True
)

print("ONNX 모델 변환 완료")