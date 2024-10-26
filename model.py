import os
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from collections import deque, Counter
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

class DataStreamBuffer:
    """Buffer for streaming data processing"""
    def __init__(self, window_size: int = 3):
        """
        Initialize data buffer.
        
        Args:
            window_size: Number of time points to keep in buffer
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def add(self, data: Dict[str, Any]) -> None:
        """Add new data point to buffer"""
        self.buffer.append(data)
        
    def is_ready(self) -> bool:
        """Check if buffer has reached required window size"""
        return len(self.buffer) == self.window_size
        
    def get_window(self) -> List[Dict[str, Any]]:
        """Get current data window"""
        return list(self.buffer)
    
    def clear(self) -> None:
        """Clear buffer contents"""
        self.buffer.clear()

class TimeSeriesPredictor:
    """Predictor for environmental and spectral time series data"""
    def __init__(self, seasonality: int = 12):
        """
        Initialize time series predictor.
        
        Args:
            seasonality: Number of time steps in seasonal cycle
        """
        self.seasonality = seasonality
        self.model = LinearRegression()
        
    def _create_features(self, timestamps: List[Any]) -> np.ndarray:
        """
        Create features for time series prediction.
        
        Args:
            timestamps: List of timestamp values
            
        Returns:
            Array of features including trend and seasonality
        """
        time_idx = np.arange(len(timestamps))
        seasonal_idx = np.sin(2 * np.pi * time_idx / self.seasonality)
        return np.column_stack([time_idx, seasonal_idx])
    
    def fit_predict(self, values: np.ndarray, timestamps: List[Any], 
                   future_steps: int) -> np.ndarray:
        """
        Fit model and predict future values.
        
        Args:
            values: Historical values
            timestamps: Corresponding timestamps
            future_steps: Number of steps to predict
            
        Returns:
            Array of predicted values
        """
        if len(values) < 2:
            return np.repeat(values[-1], future_steps)
            
        X = self._create_features(timestamps)
        y = np.array(values)
        
        # Handle NaN values
        mask = ~np.isnan(y)
        if not np.any(mask):
            return np.repeat(np.nanmean(values), future_steps)
        
        self.model.fit(X[mask], y[mask])
        
        # Create future time points
        future_idx = np.arange(len(timestamps), len(timestamps) + future_steps)
        future_seasonal = np.sin(2 * np.pi * future_idx / self.seasonality)
        X_future = np.column_stack([future_idx, future_seasonal])
        
        predictions = self.model.predict(X_future)
        
        # Ensure predictions stay within historical bounds
        min_val, max_val = np.nanmin(values), np.nanmax(values)
        predictions = np.clip(predictions, min_val, max_val)
        
        return predictions

class ModelEvaluator:
    """Helper class for model evaluation and validation"""
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                           class_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate classification predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_labels: Optional list of class names
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if class_labels is None:
            class_labels = [str(i) for i in range(max(np.max(y_true), np.max(y_pred)) + 1)]
            
        # Calculate metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, 
                                    target_names=class_labels, 
                                    output_dict=True)
        
        # Calculate per-class accuracy
        class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'class_accuracy': class_accuracy,
            'overall_accuracy': report['accuracy']
        }
    
    @staticmethod
    def validate_predictions(predictions: List[Dict[str, Any]], 
                           historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate prediction consistency and realism.
        
        Args:
            predictions: List of prediction dictionaries
            historical_data: List of historical data dictionaries
            
        Returns:
            Dictionary containing validation metrics
        """
        validation_results = {
            'temporal_consistency': True,
            'spatial_consistency': True,
            'value_ranges': True,
            'warnings': []
        }
        
        # Check temporal consistency
        dates = [pred['date'] for pred in predictions]
        if not all(dates[i] < dates[i+1] for i in range(len(dates)-1)):
            validation_results['temporal_consistency'] = False
            validation_results['warnings'].append("Dates are not strictly increasing")
        
        # Check value ranges for spectral indices
        for pred in predictions:
            for index in ['NDVI', 'NDBI', 'NDWI']:
                if np.any(pred[index] < -1) or np.any(pred[index] > 1):
                    validation_results['value_ranges'] = False
                    validation_results['warnings'].append(
                        f"Invalid {index} values detected"
                    )
        
        # Check spatial consistency
        for i in range(len(predictions)-1):
            change = np.sum(predictions[i+1]['classification'] != 
                          predictions[i]['classification']) / predictions[i]['classification'].size
            if change > 0.5:  # More than 50% change between consecutive predictions
                validation_results['spatial_consistency'] = False
                validation_results['warnings'].append(
                    f"Large spatial change detected between steps {i} and {i+1}"
                )
        
        return validation_results

class RobustLandUsePredictionModel:
    """Main land use prediction model"""
    def __init__(self, spatial_smoothing: float = 1.0, chunk_size: int = 100, 
                 n_estimators: int = 50):
        """
        Initialize land use prediction model.
        
        Args:
            spatial_smoothing: Gaussian smoothing sigma
            chunk_size: Size of data chunks for processing
            n_estimators: Number of trees in random forest
        """
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            warm_start=True
        )
        self.scaler = None
        self.spatial_smoothing = spatial_smoothing
        self.chunk_size = chunk_size
        self.shape = None
        self.window_size = 3
        self.feature_stats = None
        self.class_distribution = None
        self.n_classes = 5
        self.n_features = None
        self.historical_data = []
        self.evaluator = ModelEvaluator()
        
    def _create_features(self, data_window: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Create features from data window"""
        current_data = data_window[-1]
        y_size, x_size = current_data['classification'].shape
        features_list = []
        labels_list = []
        
        classification = current_data['classification'].copy()
        unique_classes = np.unique(classification)
        if not np.all(np.isin(unique_classes, np.arange(self.n_classes))):
            warnings.warn(f"Some classes are missing in the data. Present classes: {sorted(unique_classes)}")
        
        for y_start in range(0, y_size, self.chunk_size):
            y_end = min(y_start + self.chunk_size, y_size)
            chunk_features = []
            
            # Base features
            smoothed = gaussian_filter(
                classification[y_start:y_end].astype(float), 
                sigma=1.0
            )
            chunk_features.append(smoothed)
            
            # Current indices and environmental data
            for feature in ['NDVI', 'NDBI', 'NDWI', 'EBBI', 'DEM', 'air_temp', 'precipitation']:
                chunk_features.append(current_data[feature][y_start:y_end])
            
            # Change in indices if previous data exists
            if len(data_window) > 1:
                prev_data = data_window[-2]
                for index in ['NDVI', 'NDBI', 'NDWI', 'EBBI']:
                    change = (current_data[index][y_start:y_end] - 
                            prev_data[index][y_start:y_end])
                    chunk_features.append(change)
            else:
                for _ in range(4):
                    chunk_features.append(
                        np.zeros_like(current_data['NDVI'][y_start:y_end])
                    )
            
            X_chunk = np.stack(chunk_features, axis=-1)
            X_reshaped = X_chunk.reshape(-1, X_chunk.shape[-1])
            y_chunk = classification[y_start:y_end].reshape(-1)
            
            if self.n_features is None:
                self.n_features = X_reshaped.shape[1]
            
            features_list.append(X_reshaped)
            labels_list.append(y_chunk)
        
        X = np.vstack(features_list)
        y = np.concatenate(labels_list)
            
        return X, y
    
    def _load_processed_data(self, processed_dir: str) -> List[Dict[str, Any]]:
        """Load data from processed pickle files"""
        if not os.path.exists(processed_dir):
            raise ValueError(f"Directory {processed_dir} does not exist")
            
        processed_data_list = []
        pkl_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('_processed.pkl')])
        
        if not pkl_files:
            raise ValueError(f"No processed data files found in {processed_dir}")
        
        for filename in pkl_files:
            filepath = os.path.join(processed_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    processed_data_list.append(data)
            except Exception as e:
                warnings.warn(f"Error loading {filename}: {str(e)}")
                continue
        
        return sorted(processed_data_list, key=lambda x: x['date'])

    def _forecast_environmental_data(self, num_steps: int) -> Dict[str, np.ndarray]:
        """Forecast future environmental and spectral indices"""
        timestamps = [d['date'] for d in self.historical_data]
        forecasts = {}
        predictor = TimeSeriesPredictor()
        
        features = ['NDVI', 'NDBI', 'NDWI', 'EBBI', 'air_temp', 'precipitation']
        
        for feature in features:
            values = [np.nanmean(d[feature]) for d in self.historical_data]
            future_values = predictor.fit_predict(values, timestamps, num_steps)
            forecasts[feature] = future_values
            
        return forecasts
    
    def _generate_future_data(self, forecasts: Dict[str, np.ndarray], 
                            step_idx: int) -> Dict[str, np.ndarray]:
        """Generate future data point using forecasts"""
        last_data = self.historical_data[-1]
        future_data = {
            'DEM': last_data['DEM'].copy(),
        }
        
        for feature in forecasts.keys():
            if feature in ['air_temp', 'precipitation']:
                future_data[feature] = np.full_like(
                    last_data[feature],
                    forecasts[feature][step_idx]
                )
            else:
                historical_values = [d[feature] for d in self.historical_data]
                historical_stack = np.stack(historical_values)
                spatial_pattern = (last_data[feature] - np.nanmean(last_data[feature]))
                future_data[feature] = (spatial_pattern + forecasts[feature][step_idx])
                
                if feature in ['NDVI', 'NDBI', 'NDWI']:
                    future_data[feature] = np.clip(future_data[feature], -1, 1)
        
        return future_data

    def fit(self, processed_dir: str) -> None:
        """Fit model using processed data"""
        print("Loading processed data and starting model training...")
        
        try:
            self.historical_data = self._load_processed_data(processed_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load processed data: {str(e)}")
        
        if not self.historical_data:
            raise ValueError("No valid processed data available for training")
        
        self.shape = self.historical_data[0]['classification'].shape
        
        pbar = tqdm(total=len(self.historical_data), desc="Training Progress")
        data_buffer = DataStreamBuffer(window_size=self.window_size)
        total_class_dist = Counter()
        
        for data in self.historical_data:
            data_buffer.add(data)
            
            if data_buffer.is_ready():
                try:
                    X, y = self._create_features(data_buffer.get_window())
                    
                    if self.scaler is None:
                        self.scaler = StandardScaler()
                        self.scaler.fit(X)
                    
                    X_scaled = self.scaler.transform(X)
                    self.rf_model.fit(X_scaled, y)
                    
                    batch_dist = Counter(y)
                    total_class_dist.update(batch_dist)
                    
                    dist_str = ' '.join(f'{k}:{v}' for k, v in batch_dist.most_common())
                    pbar.set_postfix({'Classes': dist_str})
                    
                except Exception as e:
                    warnings.warn(f"Error processing batch: {str(e)}")
            
            pbar.update(1)
        
        self.class_distribution = total_class_dist
        pbar.close()
        
        # Continuing from the fit method...
        total_samples = sum(total_class_dist.values())
        print("\nFinal class distribution:")
        for class_label in range(self.n_classes):
            count = total_class_dist.get(class_label, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"Class {class_label}: {count:,} samples ({percentage:.1f}%)")
        
        print("\nTraining completed!")

    def predict(self, data_window: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate predictions for a data window.
        
        Args:
            data_window: List of data dictionaries for prediction window
            
        Returns:
            Array of predicted classifications
        """
        if self.scaler is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            X, _ = self._create_features(data_window)
            X_scaled = self.scaler.transform(X)
            predictions = self.rf_model.predict(X_scaled)
            pred_map = predictions.reshape(self.shape)
            
            if self.spatial_smoothing > 0:
                pred_map = gaussian_filter(pred_map.astype(float), 
                                        sigma=self.spatial_smoothing)
                pred_map = pred_map.round().astype(np.int8)
            
            return pred_map
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_multiple_steps(self, num_steps: int = 3) -> List[Dict[str, Any]]:
        """
        Generate multiple prediction steps with forecasted environmental data.
        
        Args:
            num_steps: Number of future time steps to predict
            
        Returns:
            List of prediction dictionaries
        """
        if not self.historical_data:
            raise ValueError("No historical data available for predictions")
        
        print("Generating future predictions...")
        
        # Create initial window from historical data
        initial_window = self.historical_data[-self.window_size:]
        current_window = list(initial_window)
        
        # Generate environmental data forecasts
        forecasts = self._forecast_environmental_data(num_steps)
        
        predictions = []
        from tqdm.notebook import tqdm  # Use notebook version for Jupyter
        
        for step in tqdm(range(num_steps), desc="Generating Predictions", position=0, leave=True):
            try:
                # Generate future environmental and spectral data
                future_data = self._generate_future_data(forecasts, step)
                
                # Make land use prediction
                pred_map = self.predict(current_window)
                
                # Create complete prediction data structure
                last_date = current_window[-1]['date']
                next_date = last_date + timedelta(days=30)
                
                prediction = {
                    'date': next_date,
                    'date_str': next_date.strftime('%Y-%m-%d'),
                    'classification': pred_map,
                    **future_data  # Include all forecasted data
                }
                
                # Validate prediction
                if len(predictions) > 0:
                    validation = self.evaluator.validate_predictions(
                        [predictions[-1], prediction],
                        self.historical_data
                    )
                    if not all(validation.values()):
                        warnings.warn(f"Validation warnings for step {step + 1}: {validation['warnings']}")
                
                predictions.append(prediction)
                
                # Update window for next prediction
                current_window.pop(0)
                current_window.append(prediction)
                
            except Exception as e:
                warnings.warn(f"Error in prediction step {step + 1}: {str(e)}")
                continue
        
        print("\nPredictions completed!")
        return predictions

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.scaler is None:
            raise ValueError("Model has not been trained yet")
            
        model_data = {
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'spatial_smoothing': self.spatial_smoothing,
            'chunk_size': self.chunk_size,
            'shape': self.shape,
            'class_distribution': self.class_distribution,
            'n_classes': self.n_classes,
            'n_features': self.n_features,
            'window_size': self.window_size
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RobustLandUsePredictionModel':
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        model = cls(
            spatial_smoothing=model_data['spatial_smoothing'],
            chunk_size=model_data.get('chunk_size', 100)
        )
        model.rf_model = model_data['rf_model']
        model.scaler = model_data['scaler']
        model.shape = model_data['shape']
        model.class_distribution = model_data.get('class_distribution', None)
        model.n_classes = model_data.get('n_classes', 5)
        model.n_features = model_data.get('n_features', None)
        model.window_size = model_data.get('window_size', 3)
        print(f"Model loaded from {filepath}")
        return model

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores from the random forest model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not hasattr(self, 'rf_model') or self.rf_model is None:
            raise ValueError("Model has not been trained yet")
            
        feature_names = [
            'Smoothed_Classification',
            'NDVI', 'NDBI', 'NDWI', 'EBBI',
            'DEM', 'Temperature', 'Precipitation',
            'NDVI_Change', 'NDBI_Change', 'NDWI_Change', 'EBBI_Change'
        ]
        
        importance_scores = self.rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importance_scores)],
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        return importance_df

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model's configuration and training status.
        
        Returns:
            Dictionary containing model summary information
        """
        return {
            'n_classes': self.n_classes,
            'n_features': self.n_features,
            'spatial_smoothing': self.spatial_smoothing,
            'chunk_size': self.chunk_size,
            'window_size': self.window_size,
            'is_trained': self.scaler is not None,
            'n_trees': self.rf_model.n_estimators if hasattr(self, 'rf_model') else None,
            'class_distribution': dict(self.class_distribution) if self.class_distribution else None,
            'input_shape': self.shape,
            'n_historical_samples': len(self.historical_data) if self.historical_data else 0
        }
