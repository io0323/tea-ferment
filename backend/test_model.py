"""
茶葉発酵度推定モデルのテスト
"""
import unittest
import numpy as np
from PIL import Image
import io
from model import TeaFermentationModel

class TestTeaFermentationModel(unittest.TestCase):
    def setUp(self):
        """
        テストの前準備
        """
        self.model = TeaFermentationModel()
        
    def test_preprocess_image(self):
        """
        画像前処理のテスト
        """
        # テスト用の画像を作成
        img = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 前処理を実行
        processed = self.model.preprocess_image(img_byte_arr)
        
        # 結果の検証
        self.assertEqual(processed.shape, (1, 224, 224, 3))
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))
        
    def test_predict(self):
        """
        予測のテスト
        """
        # テスト用の画像を作成
        img = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 予測を実行
        result = self.model.predict(img_byte_arr, 25, 60)
        
        # 結果の検証
        self.assertIn('fermentation_degree', result)
        self.assertIn('image_contribution', result)
        self.assertIn('environment_contribution', result)
        
        # 値の範囲チェック
        self.assertTrue(0 <= result['fermentation_degree'] <= 100)
        self.assertTrue(0 <= result['image_contribution'] <= 100)
        self.assertTrue(0 <= result['environment_contribution'] <= 100)
        
    def test_predict_edge_cases(self):
        """
        境界値での予測テスト
        """
        # テスト用の画像を作成
        img = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 最小値でのテスト
        result_min = self.model.predict(img_byte_arr, 20, 0)
        self.assertTrue(0 <= result_min['fermentation_degree'] <= 100)
        
        # 最大値でのテスト
        result_max = self.model.predict(img_byte_arr, 30, 100)
        self.assertTrue(0 <= result_max['fermentation_degree'] <= 100)

if __name__ == '__main__':
    unittest.main() 