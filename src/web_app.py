"""
晨读晨练签到检测系统 - FastAPI Web服务
"""
import os
import json
import io
import base64
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys

sys.path.append(os.path.dirname(__file__))
from models.three_way_decision import ThreeWayDecision
from gradcam import GradCAM

app = FastAPI(
    title="晨读晨练签到检测系统",
    description="基于三支决策的智能签到检测API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = ['晨读', '晨跑', '异常']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PredictionResult(BaseModel):
    filename: str
    label: str
    confidence: float
    decision: str
    decision_type: str

class BatchResult(BaseModel):
    total: int
    auto_pass: int
    need_review: int
    results: List[PredictionResult]
    stats: dict

class ReviewRequest(BaseModel):
    filename: str
    label: str

model = None
three_way_decision = None
gradcam = None
alpha = 0.70
beta = 0.15

def load_model_and_thresholds():
    global model, three_way_decision, gradcam, alpha, beta

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'outputs', 'resnet18_best.pt')
    report_path = os.path.join(base_dir, 'outputs', 'evaluation_report.json')

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        if 'three_way_decision' in report:
            alpha = report['three_way_decision']['alpha']
            beta = report['three_way_decision']['beta']

    three_way_decision = ThreeWayDecision(alpha=alpha, beta=beta)
    gradcam = GradCAM(model)
    print(f"✅ 模型加载完成 | 三支决策: α={alpha:.2f} β={beta:.2f}")

@app.get("/", response_class=HTMLResponse)
async def home():
    return open(os.path.join(os.path.dirname(__file__), '..', 'templates', 'index.html'), encoding='utf-8').read()

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded", "three_way": f"α={alpha:.2f} β={beta:.2f}"}

@app.get("/stats")
async def get_stats():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(base_dir, 'outputs', 'evaluation_report.json')

    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return {
            "best_val_accuracy": report.get('best_val_accuracy', 0),
            "test_accuracy": report.get('test_accuracy', 0),
            "class_accuracy": report.get('class_accuracy', {}),
            "three_way_decision": report.get('three_way_decision', {}),
            "alpha": alpha,
            "beta": beta
        }
    return {"error": "No evaluation report found"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="只支持 JPG/PNG 图片")

    try:
        img = Image.open(file.file).convert('RGB')
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)

            normal_prob = probs[0, 0].item() + probs[0, 1].item()
            decision = three_way_decision.get_decisions(torch.tensor([normal_prob]))[0].item()

            conf, pred_idx = probs.max(1)

            if decision == 0:
                label = class_names[pred_idx.item()]
                decision_text = "自动通过"
            elif decision == 1:
                label = "异常"
                decision_text = "自动拒绝"
            else:
                label = "不确定"
                decision_text = "需要人工审核"

        return {
            "filename": file.filename,
            "label": label,
            "confidence": round(conf.item(), 4),
            "probabilities": {
                "晨读": round(probs[0, 0].item(), 4),
                "晨跑": round(probs[0, 1].item(), 4),
                "异常": round(probs[0, 2].item(), 4)
            },
            "decision": decision_text,
            "decision_type": decision,
            "normal_prob": round(normal_prob, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResult)
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    auto_pass = 0
    need_review = 0

    for file in files:
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        try:
            img = Image.open(file.file).convert('RGB')
            img_t = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_t)
                probs = torch.softmax(outputs, dim=1)

                normal_prob = probs[0, 0].item() + probs[0, 1].item()
                decision = three_way_decision.get_decisions(torch.tensor([normal_prob]))[0].item()

                conf, pred_idx = probs.max(1)

                if decision == 0:
                    label = class_names[pred_idx.item()]
                    decision_text = "自动通过"
                    auto_pass += 1
                elif decision == 1:
                    label = "异常"
                    decision_text = "自动拒绝"
                    need_review += 1
                else:
                    label = "不确定"
                    decision_text = "需要人工审核"
                    need_review += 1

                results.append(PredictionResult(
                    filename=file.filename,
                    label=label,
                    confidence=round(conf.item(), 4),
                    decision=decision_text,
                    decision_type=decision
                ))
        except Exception as e:
            results.append(PredictionResult(
                filename=file.filename,
                label="错误",
                confidence=0,
                decision=str(e),
                decision_type=-1
            ))

    stats = {
        "晨读": sum(1 for r in results if r.label == "晨读"),
        "晨跑": sum(1 for r in results if r.label == "晨跑"),
        "异常": sum(1 for r in results if r.label == "异常"),
        "不确定": sum(1 for r in results if r.label == "不确定"),
        "自动通过率": round(auto_pass / len(results) * 100, 1) if results else 0,
        "审核率": round(need_review / len(results) * 100, 1) if results else 0
    }

    return BatchResult(
        total=len(results),
        auto_pass=auto_pass,
        need_review=need_review,
        results=results,
        stats=stats
    )

@app.post("/review")
async def review(request: ReviewRequest):
    return {
        "filename": request.filename,
        "label": request.label,
        "status": "saved",
        "message": f"已将 {request.filename} 标记为 {request.label}"
    }

@app.get("/class_names")
async def get_class_names():
    return {"classes": class_names}

@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="只支持 JPG/PNG 图片")

    try:
        img = Image.open(file.file).convert('RGB')
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            pred_label = class_names[pred_idx]
            confidence = probs[0, pred_idx].item()

        heatmap, _ = gradcam.generate_cam(img_t, target_class=pred_idx)
        overlay = gradcam.generate_overlay(img, heatmap)

        import numpy as np
        overlay_rgb = np.array(overlay)[:, :, ::-1]
        _, buffer = cv2.imencode('.png', overlay_rgb)
        img_base64 = base64.b64encode(buffer).decode()

        return {
            "filename": file.filename,
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "晨读": round(probs[0, 0].item(), 4),
                "晨跑": round(probs[0, 1].item(), 4),
                "异常": round(probs[0, 2].item(), 4)
            },
            "heatmap": heatmap.tolist(),
            "overlay_image": img_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize/classes")
async def get_visualization_classes():
    return {
        "classes": [
            {"id": 0, "name": "晨读", "description": "晨读签到活动"},
            {"id": 1, "name": "晨跑", "description": "晨跑签到活动"},
            {"id": 2, "name": "异常", "description": "未检测到签到活动"}
        ]
    }

if __name__ == "__main__":
    load_model_and_thresholds()
    uvicorn.run(app, host="0.0.0.0", port=8000)
