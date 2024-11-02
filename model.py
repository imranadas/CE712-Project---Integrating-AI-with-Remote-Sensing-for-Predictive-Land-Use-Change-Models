import joblib
import warnings
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom, gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any, Tuple, Optional

class DataProcessor:
    """Helper class for data processing and validation"""
    
    @staticmethod
    def resize_array(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Resize array to target shape using nearest neighbor to preserve classes.
        
        Args:
            arr: Input array to resize
            target_shape: Desired output shape
            
        Returns:
            Resized array
        """
        if arr.shape == target_shape:
            return arr
            
        try:
            # Calculate zoom factors
            zoom_factors = (target_shape[0] / arr.shape[0], 
                          target_shape[1] / arr.shape[1])
            
            # Use nearest neighbor interpolation
            resized = zoom(arr, zoom_factors, order=0)
            
            # Ensure exact target shape
            if resized.shape != target_shape:
                resized = resized[:target_shape[0], :target_shape[1]]
                
            return resized
            
        except Exception as e:
            warnings.warn(f"Error during array resizing: {str(e)}")
            return np.full(target_shape, np.nan_to_num(arr.mean()))
    
    @staticmethod
    def validate_and_standardize(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and standardize input data format."""
        # Update required fields to match new indices
        required_fields = {
            'classification', 'NDVI', 'NDBI', 'MNDWI', 'EBBI',
            'UI', 'BSI', 'DEM', 'air_temp', 'precipitation'
        }
        
        # Handle legacy data format
        if 'NDWI' in data and 'MNDWI' not in data:
            data['MNDWI'] = data['NDWI']  # Use NDWI as MNDWI for backward compatibility
        
        # Add default values for new indices if missing
        if 'UI' not in data:
            data['UI'] = np.zeros_like(data['NDVI'])  # Default UI if missing
        if 'BSI' not in data:
            data['BSI'] = np.zeros_like(data['NDVI'])  # Default BSI if missing
            
        # Check missing fields after handling legacy format
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Get target shape from classification array
        target_shape = np.asarray(data['classification']).shape[:2]
        
        # Standardize arrays
        standardized_data = {}
        for key in required_fields:
            arr = np.asarray(data[key])
            
            # Handle different shapes
            if arr.shape != target_shape:
                if len(arr.shape) == 2:
                    standardized_data[key] = DataProcessor.resize_array(arr, target_shape)
                elif len(arr.shape) == 1:
                    standardized_data[key] = np.broadcast_to(
                        arr.reshape(-1, 1), target_shape
                    )
                elif len(arr.shape) == 3:
                    std_arr = np.mean(arr, axis=2) if arr.shape[2] > 1 else arr[:, :, 0]
                    standardized_data[key] = DataProcessor.resize_array(std_arr, target_shape)
                else:
                    standardized_data[key] = np.full(target_shape, np.nan_to_num(arr.mean()))
            else:
                standardized_data[key] = arr.copy()
        
        # Ensure integer classification
        standardized_data['classification'] = standardized_data['classification'].astype(np.int32)
        
        return standardized_data

def combine_vegetation_classes(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine vegetation classes and remap other classes.
    New mapping:
    - 0: Barren Land (unchanged)
    - 1: Vegetation (combined from 1 and 2)
    - 2: Urban (remapped from 3)
    - 3: Water (remapped from 4)
    """
    processed_data = data.copy()
    classification = data['classification'].copy()
    
    # Combine vegetation classes
    vegetation_mask = (classification == 1) | (classification == 2)
    urban_mask = classification == 3
    water_mask = classification == 4
    
    # Reset to barren land
    classification = np.zeros_like(classification, dtype=np.int32)
    
    # Apply new classes
    classification[vegetation_mask] = 1  # Combined vegetation
    classification[urban_mask] = 2      # Urban
    classification[water_mask] = 3      # Water
    
    processed_data['classification'] = classification
    return processed_data

class EnhancedLandUsePredictionModel:
    def __init__(self, spatial_smoothing: float = 1.0, window_size: int = 3,
                 n_estimators: int = 50, class_weight: str = 'balanced'):
        """Initialize land use prediction model."""
        self.spatial_smoothing = spatial_smoothing
        self.window_size = window_size
        
        # Initialize model components
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            warm_start=True
        )
        
        self.scaler = None
        self.feature_names = None
        self.observed_classes = set()
        
        # Define spectral indices to use
        self.spectral_indices = ['NDVI', 'MNDWI', 'NDBI', 'EBBI', 'UI', 'BSI']
        
        # Updated class mapping for new classification scheme
        self.class_mapping = {
            0: 'Barren Land',
            1: 'Dense Vegetation',
            2: 'Moderate Vegetation',
            3: 'Urban Areas',
            4: 'Water Bodies',
            5: 'Bare Soil',
            6: 'Mixed Urban'
        }
        
        self.data_processor = DataProcessor()
    
    def _create_features(self, data_window: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature matrix from input data."""
        current_data = data_window[-1]
        
        # Get target shape
        target_shape = current_data['classification'].shape
        n_pixels = np.prod(target_shape)
        
        # Initialize features
        # For each spectral index: base + smoothed version
        n_spectral_features = len(self.spectral_indices) * 2
        # Environmental features: DEM, air_temp, precipitation
        n_env_features = 3
        n_base_features = n_spectral_features + n_env_features
        
        # Temporal features (changes in spectral indices)
        n_temporal_features = len(self.spectral_indices) if len(data_window) > 1 else 0
        n_total_features = n_base_features + n_temporal_features
        
        X = np.zeros((n_pixels, n_total_features), dtype=np.float32)
        feature_names = []
        current_idx = 0
        
        try:
            # Add spectral indices
            for index in self.spectral_indices:
                # Base feature
                X[:, current_idx] = current_data[index].ravel()
                feature_names.append(f'mean_{index}')
                current_idx += 1
                
                # Smoothed feature
                smoothed = gaussian_filter(current_data[index], sigma=1.0)
                X[:, current_idx] = smoothed.ravel()
                feature_names.append(f'smooth_{index}')
                current_idx += 1
            
            # Add environmental features
            for env_var in ['DEM', 'air_temp', 'precipitation']:
                X[:, current_idx] = current_data[env_var].ravel()
                feature_names.append(env_var)
                current_idx += 1
            
            # Add temporal features if available
            if len(data_window) > 1:
                prev_data = data_window[-2]
                for index in self.spectral_indices:
                    change = current_data[index] - prev_data[index]
                    X[:, current_idx] = change.ravel()
                    feature_names.append(f'{index}_change')
                    current_idx += 1
            
            self.feature_names = feature_names
            return X, current_data['classification'].ravel()
            
        except Exception as e:
            print(f"Error creating features: {str(e)}")
            print(f"Available keys in current_data: {list(current_data.keys())}")
            raise
    
    def fit(self, historical_data: List[Dict[str, Any]], 
            verbose: bool = True,
            batch_size: int = 10) -> None:
        """Fit model to historical data."""
        if verbose:
            print("Processing data with combined vegetation classes...")
        
        # Combine vegetation classes in all historical data
        processed_data = [combine_vegetation_classes(d) for d in historical_data]
        
        if verbose:
            print("Starting model training...")
        
        # Initialize scaling
        first_window = processed_data[:self.window_size]
        first_window_data = [self.data_processor.validate_and_standardize(d) 
                           for d in first_window]
        X_init, _ = self._create_features(first_window_data)
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_init)
        
        # Training loop
        observed_classes = set()
        class_counts = {}
        n_samples_processed = 0
        
        for start_idx in range(0, len(processed_data) - self.window_size + 1, batch_size):
            end_idx = min(start_idx + batch_size, 
                         len(processed_data) - self.window_size + 1)
            
            if verbose:
                print(f"\nProcessing batch {start_idx//batch_size + 1}...")
            
            # Process batch
            X_batch = []
            y_batch = []
            
            for i in range(start_idx, end_idx):
                try:
                    window = processed_data[i:i + self.window_size]
                    window_data = [self.data_processor.validate_and_standardize(d) 
                                 for d in window]
                    X, y = self._create_features(window_data)
                    
                    X_scaled = self.scaler.transform(X)
                    X_batch.append(X_scaled)
                    y_batch.append(y)
                    
                    # Update class info
                    unique_classes = np.unique(y)
                    observed_classes.update(unique_classes)
                    for cls in unique_classes:
                        class_counts[cls] = class_counts.get(cls, 0) + np.sum(y == cls)
                    
                    n_samples_processed += len(y)
                    
                except Exception as e:
                    warnings.warn(f"Error processing window at index {i}: {str(e)}")
                    continue
            
            if not X_batch:
                continue
                
            # Combine batch data
            try:
                X_batch_combined = np.vstack(X_batch)
                y_batch_combined = np.concatenate(y_batch)
                
                # Update model
                if start_idx == 0:
                    self.rf_model.fit(X_batch_combined, y_batch_combined)
                else:
                    n_estimators_increment = 2
                    current_n_estimators = self.rf_model.n_estimators
                    self.rf_model.set_params(
                        n_estimators=current_n_estimators + n_estimators_increment
                    )
                    self.rf_model.fit(X_batch_combined, y_batch_combined)
                    
            except Exception as e:
                warnings.warn(f"Error fitting batch: {str(e)}")
                continue
            
            # Clean up
            del X_batch, y_batch, X_batch_combined, y_batch_combined
        
        self.observed_classes = observed_classes
        
        if verbose:
            print("\nTraining completed:")
            print(f"Samples processed: {n_samples_processed:,}")
            print(f"Observed classes: {sorted(observed_classes)}")
            
            if class_counts:
                print("\nClass distribution:")
                total_samples = sum(class_counts.values())
                for cls in sorted(class_counts.keys()):
                    count = class_counts[cls]
                    percentage = (count / total_samples) * 100
                    print(f"Class {cls} ({self.class_mapping[cls]}): "
                          f"{count:,} samples ({percentage:.1f}%)")
                    
    def predict_future(self, historical_window: List[Dict[str, Any]], 
                  steps: int = 3, 
                  verbose: bool = True) -> List[Dict[str, Any]]:
        """Generate future predictions with fixed spectral indices."""
        if verbose:
            print("Starting prediction process...")
            print(f"Historical window size: {len(historical_window)}")
            print(f"Requested prediction steps: {steps}")
            
        if not historical_window:
            raise ValueError("Historical window is empty")
        
        if len(historical_window) < self.window_size:
            raise ValueError(f"Historical window too small. Need at least {self.window_size} samples")
        
        try:
            # Process historical window to combine vegetation classes
            processed_window = [combine_vegetation_classes(d) for d in historical_window]
            
            if verbose:
                print(f"\nGenerating {steps} future predictions...")
                
            predictions = []
            current_window = list(processed_window[-self.window_size:])
            
            # Get base shape from last historical data point
            base_shape = current_window[-1]['classification'].shape
            last_data = current_window[-1]
            
            # Pre-compute environmental averages
            future_env = {
                'air_temp': np.mean([d['air_temp'] for d in current_window[-3:]]),
                'precipitation': np.mean([d['precipitation'] for d in current_window[-3:]])
            }
            
            # Pre-compute spectral indices averages per class
            spectral_indices = ['NDVI', 'MNDWI', 'NDBI', 'EBBI', 'UI', 'BSI']
            class_spectral_means = {}
            
            for index_name in spectral_indices:
                class_spectral_means[index_name] = {}
                for cls in np.unique(last_data['classification']):
                    mask = last_data['classification'] == cls
                    if np.any(mask):
                        class_spectral_means[index_name][cls] = np.mean(last_data[index_name][mask])
            
            for step in range(steps):
                try:
                    if verbose:
                        print(f"\nGenerating prediction for step {step + 1}/{steps}")
                    
                    # Prepare window data
                    window_data = [self.data_processor.validate_and_standardize(d) 
                                for d in current_window]
                    
                    # Create and scale features
                    X, _ = self._create_features(window_data)
                    if self.scaler is None:
                        raise ValueError("Model not properly trained (scaler is None)")
                    X_scaled = self.scaler.transform(X)
                    
                    if verbose:
                        print(f"Feature matrix shape: {X.shape}")
                    
                    # Generate predictions
                    pred_flat = self.rf_model.predict(X_scaled)
                    pred_proba = self.rf_model.predict_proba(X_scaled)
                    
                    # Reshape predictions
                    pred_map = pred_flat.reshape(base_shape)
                    confidence_map = np.max(pred_proba, axis=1).reshape(base_shape)
                    
                    # Apply smoothing
                    if self.spatial_smoothing > 0:
                        pred_map = gaussian_filter(pred_map.astype(float), 
                                                sigma=self.spatial_smoothing)
                        pred_map = np.round(pred_map).astype(int)
                    
                    # Create prediction
                    last_date = current_window[-1]['date']
                    next_date = last_date + timedelta(days=30)
                    
                    prediction = {
                        'date': next_date,
                        'date_str': next_date.strftime('%Y-%m-%d'),
                        'classification': pred_map.copy(),
                        'confidence': confidence_map.copy(),
                        'DEM': current_window[-1]['DEM'].copy(),
                        'air_temp': np.full_like(current_window[-1]['air_temp'], 
                                            future_env['air_temp']),
                        'precipitation': np.full_like(current_window[-1]['precipitation'], 
                                                future_env['precipitation']),
                        'type': 'Predicted'
                    }
                    
                    # Calculate spectral indices for prediction using pre-computed means
                    for index_name in spectral_indices:
                        new_index = np.zeros_like(last_data[index_name])
                        means = class_spectral_means[index_name]
                        default_value = np.mean(list(means.values())) if means else 0
                        
                        for cls in np.unique(pred_map):
                            mask = pred_map == cls
                            new_index[mask] = means.get(cls, default_value)
                        
                        prediction[index_name] = new_index
                    
                    if verbose:
                        print(f"Generated prediction for {prediction['date_str']}")
                        print(f"Unique predicted classes: {np.unique(pred_map)}")
                        print(f"Mean confidence: {np.mean(confidence_map):.3f}")
                    
                    predictions.append(prediction)
                    
                    # Update window for next prediction
                    current_window.pop(0)
                    current_window.append(prediction)
                    
                except Exception as e:
                    print(f"Error in prediction step {step + 1}: {str(e)}")
                    if verbose:
                        print("\nDebug information:")
                        print(f"Feature matrix shape: {X.shape if 'X' in locals() else 'None'}")
                        if 'pred_proba' in locals():
                            print(f"Probability shape: {pred_proba.shape}")
                        print(f"Model classes: {self.rf_model.classes_}")
                    continue
            
            if not predictions and verbose:
                print("Warning: No predictions were generated successfully")
                
            return predictions
            
        except Exception as e:
            print(f"Error in prediction process: {str(e)}")
            return []
    
    def plot_prediction_results(self, predictions):
        """
        Plot prediction results with enhanced error handling.
        
        Args:
            predictions: List of prediction dictionaries
        """
        if not predictions:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No predictions available', 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.show()
            print("Warning: No predictions to plot")
            return
        
        try:
            n_predictions = len(predictions)
            fig = plt.figure(figsize=(15, 5 * n_predictions))
            gs = plt.GridSpec(n_predictions, 2, figure=fig)
            
            # Create custom colormap
            colors = ['#FFD700', '#006400', '#90EE90', '#FF0000', 
                    '#00008B', '#DEB887', '#FA8072']
            class_cmap = ListedColormap(colors)
            
            for idx, pred in enumerate(predictions):
                # Plot classification
                ax1 = fig.add_subplot(gs[idx, 0])
                im1 = ax1.imshow(pred['classification'], cmap=class_cmap)
                ax1.set_title(f"Predicted Land Use\n{pred['date_str']}")
                plt.colorbar(im1, ax=ax1, 
                            ticks=range(len(self.class_mapping)),
                            label='Class')
                
                # Plot confidence
                ax2 = fig.add_subplot(gs[idx, 1])
                if 'confidence' in pred:
                    im2 = ax2.imshow(pred['confidence'], cmap='RdYlGn')
                    ax2.set_title(f"Prediction Confidence\n{pred['date_str']}")
                    plt.colorbar(im2, ax=ax2, label='Confidence Score')
                else:
                    ax2.text(0.5, 0.5, 'Confidence data not available',
                            ha='center', va='center')
                    ax2.axis('off')
                
                # Add class distribution text
                if 'classification' in pred:
                    unique, counts = np.unique(pred['classification'], 
                                            return_counts=True)
                    total = counts.sum()
                    dist_text = "Class Distribution:\n"
                    for cls, count in zip(unique, counts):
                        percentage = (count / total) * 100
                        cls_name = self.class_mapping.get(cls, f"Class {cls}")
                        dist_text += f"{cls_name}: {percentage:.1f}%\n"
                    ax1.text(1.02, 0.5, dist_text, transform=ax1.transAxes, 
                            bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Error plotting predictions: {str(e)}', 
                    ha='center', va='center', fontsize=12, color='red')
            plt.axis('off')
            plt.show()
            print(f"Error details: {str(e)}")
    
    def plot_prediction(self, prediction=None):
        """
        Plot a single prediction with all components and error handling.
        
        Args:
            prediction: Dictionary containing prediction data, or None
        """
        if prediction is None:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No prediction data available', 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.show()
            print("Warning: No prediction data provided")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # Create custom colormap for classifications
            colors = ['#FFD700', '#006400', '#90EE90', '#FF0000', 
                    '#00008B', '#DEB887', '#FA8072']
            class_cmap = ListedColormap(colors)
            
            # Plot classification
            im1 = axes[0, 0].imshow(prediction['classification'], cmap=class_cmap)
            axes[0, 0].set_title('Land Use Classification')
            cbar1 = plt.colorbar(im1, ax=axes[0, 0], 
                            ticks=range(len(self.class_mapping)))
            cbar1.set_ticklabels(self.class_mapping.values())
            
            # Plot confidence if available
            if 'confidence' in prediction:
                im2 = axes[0, 1].imshow(prediction['confidence'], cmap='RdYlGn')
                axes[0, 1].set_title('Prediction Confidence')
                plt.colorbar(im2, ax=axes[0, 1])
            else:
                axes[0, 1].text(0.5, 0.5, 'Confidence data not available',
                            ha='center', va='center')
                axes[0, 1].axis('off')
            
            # Plot NDVI
            if 'NDVI' in prediction:
                im3 = axes[1, 0].imshow(prediction['NDVI'], cmap='RdYlGn')
                axes[1, 0].set_title('NDVI')
                plt.colorbar(im3, ax=axes[1, 0])
            else:
                axes[1, 0].text(0.5, 0.5, 'NDVI data not available',
                            ha='center', va='center')
                axes[1, 0].axis('off')
            
            # Plot NDBI
            if 'NDBI' in prediction:
                im4 = axes[1, 1].imshow(prediction['NDBI'], cmap='RdYlBu_r')
                axes[1, 1].set_title('NDBI')
                plt.colorbar(im4, ax=axes[1, 1])
            else:
                axes[1, 1].text(0.5, 0.5, 'NDBI data not available',
                            ha='center', va='center')
                axes[1, 1].axis('off')
            
            plt.suptitle(f"Prediction for {prediction.get('date_str', 'Unknown Date')}")
            plt.tight_layout()
            
            # Add stats table below the plots
            stats_text = f"Prediction Statistics:\n"
            if 'classification' in prediction:
                unique, counts = np.unique(prediction['classification'], 
                                        return_counts=True)
                total_pixels = prediction['classification'].size
                stats_text += "\nLand Use Distribution:\n"
                for cls, count in zip(unique, counts):
                    percentage = (count / total_pixels) * 100
                    cls_name = self.class_mapping.get(cls, f"Class {cls}")
                    stats_text += f"{cls_name}: {percentage:.1f}%\n"
            
            if 'confidence' in prediction:
                stats_text += f"\nConfidence Metrics:\n"
                stats_text += f"Mean: {np.mean(prediction['confidence']):.3f}\n"
                stats_text += f"Min: {np.min(prediction['confidence']):.3f}\n"
                stats_text += f"Max: {np.max(prediction['confidence']):.3f}\n"
            
            plt.figtext(0.1, -0.1, stats_text, fontsize=10, va='top')
            plt.subplots_adjust(bottom=0.2)
            
            plt.show()
            
        except Exception as e:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Error plotting prediction: {str(e)}', 
                    ha='center', va='center', fontsize=12, color='red')
            plt.axis('off')
            plt.show()
            print(f"Error details: {str(e)}")
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'spatial_smoothing': self.spatial_smoothing,
            'window_size': self.window_size,
            'feature_names': self.feature_names,
            'observed_classes': self.observed_classes,
            'class_mapping': self.class_mapping
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EnhancedLandUsePredictionModel':
        """Load model from file."""
        model_data = joblib.load(filepath)
        model = cls(
            spatial_smoothing=model_data['spatial_smoothing'],
            window_size=model_data['window_size']
        )
        model.rf_model = model_data['rf_model']
        model.scaler = model_data['scaler']
        model.feature_names = model_data['feature_names']
        model.observed_classes = model_data['observed_classes']
        model.class_mapping = model_data['class_mapping']
        print(f"Model loaded from: {filepath}")
        return model