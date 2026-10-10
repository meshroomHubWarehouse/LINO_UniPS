from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything 
import argparse
import os
import torch
def predict_normal():
    test_loader = DataLoader(testdata, batch_size=1)
    trainer = pl.Trainer(accelerator="auto", devices=1,precision="bf16-mixed")
    trainer.test(model=lino, dataloaders=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", 
        type=str, 
        default="DiLiGenT", 
        help="Name of the task"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data/DiLiGenT/",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=16,
        help="Number of images to process"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional path to a local lino.pth checkpoint. If omitted, LINO_MODEL_URL/default URL is used.",
    )

    
    args = parser.parse_args()
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    seed_everything(seed=args.seed, workers=True)
    lino = torch.hub.load(
            repo_dir,
            "lino_unips",
            source="local",
            pretrained=True,
            task_name=args.task_name,
            ckpt_path=args.ckpt_path,
        )
    testdata = torch.hub.load(
            repo_dir,
            "load_test_data",
            source="local",
            data_root=[args.data_root],
            numofimages=args.num_images
        )
    predict_normal()
