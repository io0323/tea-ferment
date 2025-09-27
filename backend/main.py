"""
茶葉発酵度推定API

茶葉の画像と環境データ（温度・湿度）から発酵度を推定するFastAPIアプリケーション
"""
import io
import json
import logging

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model import TeaFermentationModel

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPIアプリケーションの作成
app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # フロントエンドのURLを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 共通のレスポンスヘッダーを設定
@app.middleware("http")
async def add_cors_headers(request, call_next):
    """
    CORSヘッダーを追加するミドルウェア
    
    Args:
        request: HTTPリクエスト
        call_next: 次のミドルウェアまたはエンドポイント
    
    Returns:
        response: CORSヘッダーが追加されたレスポンス
    """
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.get("/")
async def root():
    """
    ルートエンドポイント
    
    Returns:
        dict: APIの説明メッセージ
    """
    return {"message": "茶葉発酵度推定API"}

@app.get("/health")
async def health_check():
    """
    ヘルスチェックエンドポイント
    
    Returns:
        dict: サーバーの状態
    """
    return {"status": "healthy"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...)
):
    """
    茶葉の発酵度を推定するAPIエンドポイント
    
    Args:
        image: 茶葉の画像ファイル
        temperature: 温度（°C）
        humidity: 湿度（%）
    
    Returns:
        dict: 推定結果（発酵度と寄与度）
    """
    try:
        logger.info("Received prediction request")
        logger.info("Temperature: %s°C, Humidity: %s%%", temperature, humidity)
        logger.info("Image filename: %s, Content type: %s", image.filename, image.content_type)
        
        # 画像の読み込みと前処理
        try:
            contents = await image.read()
            logger.info("Image size: %s bytes", len(contents))
            img = Image.open(io.BytesIO(contents))
            img = img.resize((224, 224))
            logger.info("Processed image size: %s", img.size)
        except (IOError, OSError, ValueError) as e:
            logger.error("Image processing error: %s", str(e))
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid image format: {str(e)}"},
                headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
            )
        
        # モデルの初期化と予測
        try:
            model = TeaFermentationModel()
            logger.info("Model initialized")
            
            # 予測の実行
            result = model.predict(contents, temperature, humidity)
            logger.info("Prediction result: %s", result)
            return result
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error("Model prediction error: %s", str(e))
            return JSONResponse(
                status_code=500,
                content={"detail": f"Model prediction error: {str(e)}"},
                headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
            )
        
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Unexpected error in predict endpoint: %s", str(e))
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected error: {str(e)}"},
            headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
        )

if __name__ == "__main__":
    try:
        logger.info("Starting server...")
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="debug",
            workers=1
        )
    except (OSError, RuntimeError) as e:
        logger.error("Failed to start server: %s", str(e))
        raise

