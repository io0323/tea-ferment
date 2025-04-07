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
    sensor_data: str = Form(...)
):
    """
    茶葉の発酵度を推定するAPIエンドポイント
    
    Args:
        image: 茶葉の画像ファイル
        sensor_data: センサーデータ（JSON形式）
    
    Returns:
        dict: 推定結果（発酵度とレベル）
    """
    try:
        logger.info("Received prediction request")
        
        # センサーデータのパース
        try:
            sensor_dict = json.loads(sensor_data)
            logger.info(f"Parsed sensor data: {sensor_dict}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data: {sensor_data}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid sensor data format"},
                headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
            )
        
        # 画像の読み込みと前処理
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            img = img.resize((224, 224))
            logger.info(f"Processed image size: {img.size}")
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid image format"},
                headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
            )
        
        # ダミーの推論結果を返す
        fermentation_score = np.random.uniform(0, 1)
        
        # 発酵レベルの判定
        if fermentation_score < 0.3:
            level = "未発酵"
        elif fermentation_score < 0.7:
            level = "中発酵"
        else:
            level = "高発酵"
        
        response = {
            "fermentation_score": float(fermentation_score),
            "fermentation_level": level,
            "temperature": sensor_dict["temperature"],
            "humidity": sensor_dict["humidity"]
        }
        logger.info(f"Sending response: {response}")
        return JSONResponse(
            content=response,
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
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