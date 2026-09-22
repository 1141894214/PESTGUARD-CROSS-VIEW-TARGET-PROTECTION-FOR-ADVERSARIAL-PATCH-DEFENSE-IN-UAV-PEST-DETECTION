"""Evaluate saved PestGuard JSON outputs against YOLO-format labels.

This does not generate attacks or reconstruct the manuscript's unpublished splits.
Use the exact evaluated manifest and attack samples for a paper comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from pestguard.core import Detections
from pestguard.evaluation import EvalRecord, ap50_101


def load_yolo(path: Path, width: int, height: int, offset: int):
    boxes=[];labels=[]
    if not path.exists():
        raise FileNotFoundError(f"missing label file (use an empty file for no-target images): {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        parts=line.split()
        if len(parts)<5:continue
        cls=int(parts[0])+offset
        cx,cy,bw,bh=map(float,parts[1:5])
        boxes.append([(cx-bw/2)*width,(cy-bh/2)*height,
                      (cx+bw/2)*width,(cy+bh/2)*height])
        labels.append(cls)
    return torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),torch.tensor(labels,dtype=torch.long)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",required=True,help="Directory containing result.json files")
    p.add_argument("--labels",required=True,help="YOLO .txt label directory")
    p.add_argument("--class-ids",required=True,nargs="+",type=int,
                   help="Detector class IDs to evaluate")
    p.add_argument("--label-class-offset",type=int,default=0,
                   help="Add to YOLO label IDs; torchvision foreground often needs +1")
    p.add_argument("--prediction-key",choices=["detections_original","detections_final"],
                   default="detections_final")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    paths=sorted(Path(args.results).rglob("result.json"))
    if not paths:raise SystemExit("no result.json files found")
    records=[]
    for path in paths:
        data=json.loads(path.read_text(encoding="utf-8"))
        input_path=Path(data["input"])
        if not input_path.exists():raise FileNotFoundError(input_path)
        width,height=Image.open(input_path).size
        gt_boxes,gt_labels=load_yolo(Path(args.labels)/(input_path.stem+".txt"),width,height,
                                       args.label_class_offset)
        pred=data[args.prediction_key]
        d=Detections(torch.tensor([v["box_xyxy"] for v in pred],dtype=torch.float32).reshape(-1,4),
                     torch.tensor([v["score"] for v in pred],dtype=torch.float32),
                     torch.tensor([v["class_id"] for v in pred],dtype=torch.long))
        records.append(EvalRecord(str(input_path.resolve()),gt_boxes,gt_labels,d))
    metrics=ap50_101(records,args.class_ids)
    metrics["protocol_note"]="Proposed 101-point AP50; verify against the original manuscript evaluator."
    metrics["prediction_key"]=args.prediction_key
    metrics["label_class_offset"]=args.label_class_offset
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
