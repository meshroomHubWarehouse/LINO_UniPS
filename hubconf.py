import os
import numpy as np
from typing import Optional
import torch
from src.models import LiNo_UniPS 
from src.data import TestData
from src.data import DemoData
dependencies = ['torch', 'pytorch_lightning', 'numpy']

DEFAULT_MODEL_URL = os.getenv(
    "LINO_MODEL_URL",
    "https://huggingface.co/houyuanchen/lino/resolve/main/lino.pth",
)
# PBR model (normals + base color, roughness, metallic: src.models.LiNo_UniPS_PBR with brdf=True)
DEFAULT_PBR_MODEL_URL = os.getenv(
    "LINO_PBR_MODEL_URL",
    "https://huggingface.co/houyuanchen/lino/resolve/main/lino_pbr.pth",
)

def lino_unips(pretrained=True, task_name="DiLiGenT", ckpt_path: Optional[str] = None, **kwargs):
    model = LiNo_UniPS(task_name=task_name, **kwargs)
    if pretrained:
        state_dict = _load_state_dict(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("load lino_unips successfully")
            
    return model

def load_test_data(data_root: list, numofimages: int):
    return TestData(data_root,numofimages)

def load_data(input_imgs_list, input_mask, **data_options):
    return DemoData(input_imgs_list,input_mask, **data_options)

def LINO(local_file_path: Optional[str] = None,task_name="DiLiGenT", **kwargs):
    """
    Load the LINO model with optional local file path for state_dict.
    
    Args:
        local_file_path (str, optional): Path to the local state_dict file. If None, uses the default URL.
        
    Returns:
        Predictor: An instance of the Predictor class with the loaded model.
    """
    state_dict = _load_state_dict(local_file_path)
    model = LiNo_UniPS(task_name=task_name, **kwargs)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return Predictor(model)


def _load_state_dict(local_file_path: Optional[str] = None, url: Optional[str] = None):
    """State dict from local_file_path if given, otherwise downloaded from url (DEFAULT_MODEL_URL by default)."""
    url = DEFAULT_MODEL_URL if url is None else url
    if local_file_path is not None:
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {local_file_path}")
        # Load state_dict from local file
        state_dict = torch.load(local_file_path, weights_only=False, map_location=torch.device("cpu"))
    else:
        if not url:
            raise ValueError("LINO_MODEL_URL environment variable is not set; please provide a local_file_path or set LINO_MODEL_URL.")
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location=torch.device("cpu"))

    return state_dict.get("state_dict", state_dict) if isinstance(state_dict, dict) else state_dict


class Predictor:
    def __init__(self, model, device=None):
        self.model = model
        # GPU by default (bf16 on compute capability >= 8, fp16 otherwise), bf16 on CPU: the network casts
        # its intermediate tensors to bf16 (e.g. the wavelet filters), float32 weights would not match them
        self.device = torch.device(device) if device else torch.device('cuda')
        if self.device.type == 'cuda':
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
        else:
            self.dtype = torch.bfloat16

        self.model.to(self.device, dtype=self.dtype)
    
    def predict(self, input_imgs_list, input_mask, **data_options):
        demodata = load_data(input_imgs_list, input_mask, **data_options)
        data = demodata[0]
        for key in data:
            # Only floating-point data is cast to the inference dtype: integer data (e.g. the crop box "roi")
            # must stay exact, bf16 would round it (1595 -> 1592) and shift the predicted normals.
            if isinstance(data[key], np.ndarray):
                dtype = self.dtype if np.issubdtype(data[key].dtype, np.floating) else torch.long
                data[key] = torch.tensor(data[key], device=self.device, dtype=dtype)[None, ...]  # Add None to keep the batch dimension
            elif isinstance(data[key], torch.Tensor):
                dtype = self.dtype if torch.is_floating_point(data[key]) else torch.long
                data[key] = data[key].to(self.device, dtype=dtype)[None, ...]
            elif data[key] is None:
                data[key] = None
            else:
                raise TypeError(f"Unsupported data type: {type(data[key])}")

        with torch.no_grad():
            output = self.model(data)
        return output
    
    