"""Run PestGuard on one image or an image directory using real checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from pestguard import DefenseConfig, PestGuard
from pestguard.adapters import (
    TorchvisionFasterRCNN, UltralyticsYOLO, HuggingFaceDETR,
    load_vgg19_style_extractor,
)
from pestguard.io import load_rgb, save_result


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,help="Image or directory of PNG/JPEG/TIFF inputs")
    p.add_argument("--output",required=True,help="Output directory")
    p.add_argument("--detector",required=True,choices=["fasterrcnn","yolo","detr"])
    p.add_argument("--detector-checkpoint",required=True,
                   help="Trained detector checkpoint or local HuggingFace model directory")
    p.add_argument("--vgg-checkpoint",required=True,
                   help="Explicit VGG19 checkpoint, ImageNet or PDT-fine-tuned; no auto-download")
    p.add_argument("--num-classes",type=int,
                   help="Faster R-CNN only: includes background class")
    p.add_argument("--config",help="JSON overrides for DefenseConfig proposed defaults")
    p.add_argument("--device",default="auto",help="cpu, cuda:0, or auto")
    p.add_argument("--max-images",type=int,default=0,help="0 means all")
    return p.parse_args()


def main() -> None:
    args=parse_args()
    device=("cuda:0" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    overrides=json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if "target_class_ids" in overrides and overrides["target_class_ids"] is not None:
        overrides["target_class_ids"]=tuple(overrides["target_class_ids"])
    cfg=DefenseConfig(**overrides)
    if args.detector=="fasterrcnn":
        if args.num_classes is None:
            raise SystemExit("--num-classes is required for Faster R-CNN")
        detector=TorchvisionFasterRCNN(args.detector_checkpoint,args.num_classes,device)
    elif args.detector=="yolo":
        detector=UltralyticsYOLO(args.detector_checkpoint,device)
    else:
        detector=HuggingFaceDETR(args.detector_checkpoint,device)
    extractor=load_vgg19_style_extractor(args.vgg_checkpoint,device)
    pipeline=PestGuard(detector,extractor,cfg)
    inp=Path(args.input)
    if inp.is_file():
        images=[inp]
    else:
        images=sorted(p for p in inp.rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"})
    if args.max_images>0:images=images[:args.max_images]
    if not images:raise SystemExit("no input images found")
    out=Path(args.output)
    for idx,path in enumerate(images,1):
        print(f"[{idx}/{len(images)}] {path}",flush=True)
        x=load_rgb(path,device)
        result=pipeline.run(x)
        target=out/(path.stem if len(images)>1 else "result")
        save_result(result,target,str(path),asdict(cfg))
    print(f"Saved {len(images)} result(s) to {out}")


if __name__=="__main__":main()
