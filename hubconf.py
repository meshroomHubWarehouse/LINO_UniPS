import os
import numpy as np
from typing import Optional
import torch
from src.models import LiNo_UniPS 
from src.data import TestData
from src.data import DemoData
dependencies = ['torch', 'pytorch_lightning', 'numpy']

DEFAULT_MODEL_URL = os.getenv("LINO_MODEL_URL", "")  

def lino_unips(pretrained=True, task_name="DiLiGenT", **kwargs):
    model = LiNo_UniPS(task_name=task_name, **kwargs)
    if pretrained:
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL,
                progress=True
            )
            model.load_state_dict(state_dict)
            model.eval()
            print("load lino_unips successfully")
        except Exception as e:
            print(f"error{e}")
            
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


def _load_state_dict(local_file_path: Optional[str] = None):
    if local_file_path is not None and os.path.exists(local_file_path):
        # Load state_dict from local file
        state_dict = torch.load(local_file_path, weights_only=False, map_location=torch.device("cpu"))
    else:
        if not DEFAULT_MODEL_URL:
            raise ValueError("LINO_MODEL_URL environment variable is not set; please provide a local_file_path or set LINO_MODEL_URL.")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True, map_location=torch.device("cpu"))

    return state_dict


class Predictor:
    def __init__(self, model, device=None):
        self.model = model
        # GPU by default (bf16 on compute capability >= 8, fp16 otherwise), float32 on CPU
        self.device = torch.device(device) if device else torch.device('cuda')
        if self.device.type == 'cuda':
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

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
    
    