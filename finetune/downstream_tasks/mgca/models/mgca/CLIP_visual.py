import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn
import utils_builder
import os.path as osp
import os 
from argparse import ArgumentParser
import random
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from torchvision.transforms import transforms

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img, attn_map, blur=True):
#     axes = plt.plot(figsize=(5, 5))
#     axes[0].imshow(img)
    plt.figure(figsize=(5,5))
    plt.imshow(getAttMap(img, attn_map, blur))
    
    plt.axis("off")
    plt.show()
    save_path = osp.join(os.path.abspath('.'), "save_attn_visual/img_attn_64644_sp.svg")
    plt.savefig(save_path, bbox_inches = 'tight')
    
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
     proj_v:nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output = proj_v(output)
        
        print('output shape', output.shape)
        print('target shape', target.shape)
        output.backward(target) # .shape
#         print('output', output.backward(target))

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    
    print('the original input shape: ', input.shape)
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam

def preprocess(img):
    
    image = Image.open(img).convert('RGB')
    image = image.resize((500, 500))
    image = np.asarray(image).astype(np.float32) / 255.
    
    # Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
    trans = transforms.Compose([
         transforms.ToTensor(),
         normalize, 
    ])

    img = trans(image)
    return img

if __name__ == '__main__':
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(2020)
    np.random.seed(2020)
    
    parser = ArgumentParser()
    
    
    image_caption = 'Cardiomegaly'
    clip_model = "RN50" 
    saliency_layer = "layer4" #@param ["layer4", "layer3", "layer2", "layer1"]

    blur = True 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load(clip_model, device=device, jit=False)
    
    # start to loading  ##############################################
    parser.add_argument('--lm_model', type=str, required=False, default='/cxr_bert')
    
    parser.add_argument('--vision_model', type=str, required=False, default='/cxr_bert')
    
    args = parser.parse_args() # pretrain_models/pretrain_mmodal_TF_TT_SLIP_free_9/epoch80/cross-lingual_multi-modal_encoder.pth
    
    args.vision_model = osp.join(os.path.abspath('.'), "pretrain_models/pretrain_mmodal_TF_TT_SLIP_free_9/epoch80/cross-lingual_multi-modal_encoder.pth")
    args.lm_model = osp.join(os.path.abspath('.'), "zero-shot/lm_model/") 
    
    model = utils_builder.ResNet_CXRBert(args)
    model_path = osp.join(os.path.abspath('.'), "zero-shot/Camp_models/tf_slip_loss/epoch50/cross-lingual_multi-modal_total.pth") # tf_slip_loss
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    
    model = model.cuda()
    
    print('loading total checkpoint ok !')
    model.eval()
    
    ##############################################################################
    # Put the image here:
    image_path = osp.join(os.path.abspath('.'), "ChexpertLocalize/cheXlocalize/localize_data/CheXpert/val/patient64644/study1/view1_frontal.jpg")
    image_input = preprocess(image_path).unsqueeze(0).to(device)
    image_np = load_image(image_path, 500)
    texts = model._tokenize([image_caption]).to(device)
    
    ## 
#     image_input = model.proj_v(image_input)
    
    text_input = model.lm_model(texts.input_ids.to(device=torch.device("cuda" if torch.cuda.is_available() else"cpu")) 
                                              , texts.attention_mask.to(device=torch.device("cuda"if torch.cuda.is_available()else"cpu")) 
                                             ).last_hidden_state[:, 0]
    
    text_input = model.proj_t(text_input)

    attn_map = gradCAM(
        model.encoder,
        model.proj_v,
        image_input,
        # model.encode_text(text_input).float(),
        text_input, # target
        getattr(model.encoder, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()
    print('attn_map shape',attn_map.shape )

    viz_attn(image_np, attn_map, blur)