"""
茶葉発酵度推定モデル
"""
import io

import numpy as np
import tensorflow as tf
from PIL import Image


class TeaFermentationModel:
    """
    茶葉発酵度推定モデルクラス

    画像と環境データから茶葉の発酵度を推定する
    """

    def __init__(self):
        """
        モデルの初期化
        """
        self.model = self._build_model()
        self.image_size = (224, 224)

    def _build_model(self):
        """
        モデルの構築
        """
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def preprocess_image(self, image_bytes):
        """
        画像の前処理
        """
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize(self.image_size)
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_bytes, temperature, humidity):
        """
        発酵度の予測
        """
        # 画像の前処理
        processed_image = self.preprocess_image(image_bytes)
        # 環境データの正規化
        normalized_temp = (temperature - 20) / 10  # 20-30℃の範囲を0-1に正規化
        normalized_humidity = humidity / 100  # 0-100%の範囲を0-1に正規化
        # 予測
        image_pred = self.model.predict(processed_image)[0][0]
        # 環境データの重み付け
        env_factor = (normalized_temp + normalized_humidity) / 2
        # 最終的な発酵度の計算（0-100%）
        final_prediction = (image_pred * 0.7 + env_factor * 0.3) * 100
        return {
            'fermentation_degree': float(final_prediction),
            'image_contribution': float(image_pred * 100),
            'environment_contribution': float(env_factor * 100)
        }
