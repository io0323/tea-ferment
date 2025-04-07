# 茶葉発酵度推定システム

茶葉の発酵度を画像と環境データ（温度・湿度）から推定するWebアプリケーションです。

## 機能

- 茶葉の画像アップロード
- 温度・湿度データの入力
- 発酵度の推定と可視化
- 推定結果の表示（発酵度スコアとレベル）

## 技術スタック

### フロントエンド
- Next.js 14
- TypeScript
- Tailwind CSS
- React Hook Form
- Axios
- Recharts

### バックエンド
- FastAPI
- Python 3.9
- Uvicorn
- Pillow (PIL)

## セットアップ

### バックエンド

1. バックエンドディレクトリに移動:
```bash
cd backend
```

2. 仮想環境を作成して有効化:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 依存パッケージをインストール:
```bash
pip install -r requirements.txt
```

4. サーバーを起動:
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### フロントエンド

1. フロントエンドディレクトリに移動:
```bash
cd frontend
```

2. 依存パッケージをインストール:
```bash
npm install
```

3. 開発サーバーを起動:
```bash
npm run dev
```

## APIエンドポイント

### ヘルスチェック
- `GET /health`
- レスポンス: `{"status": "healthy"}`

### 発酵度推定
- `POST /predict`
- パラメータ:
  - `image`: 茶葉の画像ファイル
  - `sensor_data`: 温度・湿度データ（JSON形式）
- レスポンス:
```json
{
  "fermentation_score": 0.75,
  "fermentation_level": "高発酵",
  "temperature": 25.5,
  "humidity": 65.0
}
```

## ライセンス

MIT License

## 作者

io0323 