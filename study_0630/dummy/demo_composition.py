import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
import vltseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

# This sets the default registry to mmseg::model (instead of mmengine::model), which is where SegDataPreProcessor is found
# This is not necessary when using the test script, since default_runtime.py includes default_scope = 'mmseg'
# In this file, we are only building the 'model' part of the config, so that information is lost
register_all_modules(True)

def rsz(input, l):
    transform = torchvision.transforms.Resize((l, l))
    return transform(input)

config = "/data/choihy/study_0630/VLTSeg/configs/custom_inference.py"
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config.fromfile(config)
model = MODELS.build(cfg.model)
model.to(device)

# Load checkpoint directly instead of using init_weights
checkpoint_path = '/data/choihy/study_0630/VLTSeg/vltseg_checkpoint_mapillary+cityscapes_1.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'], strict=False)

if device == 'cpu':
    model = revert_sync_batchnorm(model)

with torch.no_grad():
    # Load and resize image to match checkpoint size (1022x1022)
    image = read_image('images/CLIP.png', mode=ImageReadMode.RGB).to(torch.float32)
    image = rsz(image, 1022).to(device).unsqueeze(0)
    
    # Normalize image (ImageNet normalization)
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
    image = (image - mean) / std
    result = model.predict(image)
    predictions = result[0].seg_logits.data

print(str(result))


# with torch.no_grad():
#     # Load and resize image to match checkpoint size (1022x1022)
#     image = read_image('images/CLIP.png', mode=ImageReadMode.RGB).to(torch.float32)
#     image = rsz(image, 1022).to(device).unsqueeze(0)
    
#     # Normalize image (ImageNet normalization)
#     mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
#     std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
#     image = (image - mean) / std
