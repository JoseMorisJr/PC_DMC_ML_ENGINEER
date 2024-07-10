import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from joblib import dump, load
import json
from autoML import load_data, split_data, model, random_search, fit_model, print_score, save_model, running_model

class TestModelFunctions(unittest.TestCase):

     # Prueba para la función split_data
    def test_split_data(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'feature1': [10, 20, 30, 40],
            'feature2': [15, 25, 35, 45],
            'target': [0, 1, 0, 1]
        })

        X_train, X_test, y_train, y_test = split_data(df)

        self.assertEqual(X_train.shape, (2, 2))
        self.assertEqual(X_test.shape, (2, 2))
        self.assertEqual(y_train.shape, (2,))
        self.assertEqual(y_test.shape, (2,))

    # Prueba para la función model
    def test_model(self):
        model_rf = model()
        self.assertIsInstance(model_rf, RandomForestClassifier)

    # Prueba para la función random_search
    def test_random_search(self):
        model_rf = model()
        random_search_rf = random_search(model_rf)

        self.assertIsInstance(random_search_rf, RandomizedSearchCV)
        self.assertEqual(random_search_rf.cv, 5)

    # Prueba para la función fit_model
    def test_fit_model(self):
        model_rf = model()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        fit_model(model_rf, X, y)
        self.assertTrue(model_rf.fit)


    # Prueba para la ejecución del modelo
    @patch('autoML.load_data')
    @patch('autoML.split_data')
    @patch('autoML.model')
    @patch('autoML.random_search')
    @patch('autoML.fit_model')
    @patch('autoML.print_score')
    @patch('autoML.save_model')
    def test_running_model(self, mock_save_model, mock_print_score, mock_fit_model, mock_random_search, mock_model, mock_split_data, mock_load_data):
        mock_df = pd.DataFrame({'id': [1, 2], 'target': [0, 1], 'feature1': [10, 20], 'feature2': [30, 40]})
        mock_load_data.return_value = mock_df
        mock_split_data.return_value = (None, None, None, None)
        mock_model.return_value = MagicMock()
        mock_random_search.return_value = MagicMock(best_estimator_=MagicMock())
        
        running_model()

        mock_load_data.assert_called_once_with('/opt/airflow/dags/data/temp/train_temp.csv')
        mock_split_data.assert_called_once()
        mock_model.assert_called_once()
        mock_random_search.assert_called_once()
        mock_fit_model.assert_called_once()
        mock_print_score.assert_called_once()
        mock_save_model.assert_called_once()

if __name__ == '__main__':
    unittest.main()