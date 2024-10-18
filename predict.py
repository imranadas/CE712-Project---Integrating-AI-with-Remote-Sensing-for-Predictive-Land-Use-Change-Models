import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def calculate_ndvi(npy_data):
    band_nir = npy_data[3]  # Assuming Band 4 (index 3) is NIR
    band_red = npy_data[2]  # Assuming Band 3 (index 2) is Red
    ndvi = (band_nir - band_red) / (band_nir + band_red)
    return ndvi

def classify_land_use(npy_data):
    ndvi = calculate_ndvi(npy_data)
    
    # Simple thresholding for classification based on NDVI and Landsat bands
    urban_mask = (npy_data[0] > 0.3)  # Band 1 (blue) often used for urban detection
    vegetation_mask = (ndvi > 0.2)    # NDVI threshold for vegetation
    water_mask = (npy_data[1] < 0.1)  # Band 2 (green) for water body detection
    
    return urban_mask, vegetation_mask, water_mask

def predict_future(model, data_loader, start_time, end_time):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(next(model.parameters()).device)
            future_steps = end_time - start_time + 1
            
            # Use the last sequence from inputs as the starting point
            last_sequence = inputs[:, -1:, :, :, :]
            
            # Predict future steps
            for _ in range(future_steps):
                pred = model(last_sequence)
                predictions.append(pred.cpu().numpy())
                
                # Update last_sequence for next prediction
                last_sequence = torch.cat([last_sequence[:, 1:], pred.unsqueeze(1)], dim=1)
    
    # Combine all predictions
    predictions = np.concatenate(predictions, axis=1)
    return predictions

def evaluate_predictions(predictions, data_loader, start_time, end_time):
    # Get actual data for comparison
    actual_data = []
    for inputs, _ in data_loader:
        actual_data.extend([inputs[:, t, :, :, :].numpy() for t in range(start_time - inputs.shape[1], end_time + 1) if t >= 0])

    # Ensure predictions and actual_data have the same length
    predictions = predictions[:, :len(actual_data)]

    # Calculate MSE for each time step
    mse_scores = []
    for t, (pred, actual) in enumerate(zip(predictions, actual_data)):
        mse = mean_squared_error(actual.flatten(), pred.flatten())
        mse_scores.append(mse)

    # Plot MSE over time
    plt.figure(figsize=(10, 5))
    plt.plot(range(start_time, end_time + 1), mse_scores)
    plt.title('Mean Squared Error over Time')
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.show()

    # Visualize predictions vs actual for the last time step
    last_pred = predictions[-1]
    last_actual = actual_data[-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Predicted vs Actual Land Use (Last Time Step)')

    for i, (title, data) in enumerate([('Predicted', last_pred), ('Actual', last_actual)]):
        urban_mask, vegetation_mask, water_mask = classify_land_use(data)
        
        axes[i, 0].set_title(f'{title} Urban Development')
        axes[i, 0].imshow(urban_mask, cmap='gray')
        
        axes[i, 1].set_title(f'{title} Vegetation')
        axes[i, 1].imshow(vegetation_mask, cmap='Greens')
        
        axes[i, 2].set_title(f'{title} Water Bodies')
        axes[i, 2].imshow(water_mask, cmap='Blues')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # This section is for testing the prediction and evaluation functions
    # You would typically call these functions from training.py
    pass