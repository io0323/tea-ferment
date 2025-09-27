from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import json
import logging
import uvicorn

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
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.get("/")
async def root():
    return {"message": "茶葉発酵度推定API"}

@app.get("/health")
async def health_check():
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
        logger.info(f"Temperature: {temperature}°C, Humidity: {humidity}%")
        logger.info(f"Image filename: {image.filename}, Content type: {image.content_type}")
        
        # 画像の読み込みと前処理
        try:
            contents = await image.read()
            logger.info(f"Image size: {len(contents)} bytes")
            img = Image.open(io.BytesIO(contents))
            img = img.resize((224, 224))
            logger.info(f"Processed image size: {img.size}")
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid image format: {str(e)}"},
                headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
            )
        
        # モデルの初期化と予測
        try:
            from model import TeaFermentationModel
            model = TeaFermentationModel()
            logger.info("Model initialized")
            
            # 予測の実行
            result = model.predict(contents, temperature, humidity)
            logger.info(f"Prediction result: {result}")
            return result
        except Exception as e:
            logger.error(f"Model prediction error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Model prediction error: {str(e)}"},
                headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
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
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise 