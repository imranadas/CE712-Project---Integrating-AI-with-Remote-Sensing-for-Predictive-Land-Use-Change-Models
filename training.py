import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
from predict import predict_future, evaluate_predictions
from torch.cuda.amp import GradScaler, autocast

# Configuration section
config = SimpleNamespace()

# File and data loading configurations
config.data_dir = 'Data\Pre-Processed\Jaipur'
config.file_format = '{}.npy'
config.batch_size = 4 
config.shuffle = False
config.sequence_length = 1 

# Model configurations
config.input_dim = 10
config.hidden_dim = 32
config.kernel_size = 3
config.num_layers = 2
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configurations
config.num_epochs = 20
config.learning_rate = 0.001
config.loss_function = nn.MSELoss()

# Prediction configurations
config.future_prediction_start = 90
config.future_prediction_end = 100

# Model saving configuration
config.models_dir = 'Models'

# Updated Dataset to handle multiple time snapshots
class LandUseTimeSeriesDataset(Dataset):
    def __init__(self, npy_directory, sequence_length):
        self.npy_directory = npy_directory
        self.sequence_length = sequence_length
        self.files = sorted(os.listdir(npy_directory))
        self.temporal_data = self._extract_temporal_info(self.files)

    def _extract_temporal_info(self, files):
        return list(range(len(files)))

    def __len__(self):
        return max(0, len(self.files) - self.sequence_length + 1)

    def __getitem__(self, idx):
        sequence = []
        for i in range(self.sequence_length):
            npy_file = os.path.join(self.npy_directory, self.files[idx + i])
            data = np.load(npy_file)
            sequence.append(data)
        
        sequence = np.array(sequence)
        timestamp = self.temporal_data[idx:idx+self.sequence_length]
        
        return torch.from_numpy(sequence).float(), torch.tensor(timestamp)

# Define the ConvLSTM Model
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.convlstm_cells = nn.ModuleList([ConvLSTMCell(input_dim=input_dim if i == 0 else hidden_dim,
                                                          hidden_dim=hidden_dim,
                                                          kernel_size=kernel_size)
                                             for i in range(num_layers)])

        self.output_layer = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, input_tensor):
        batch_size, seq_len, _, height, width = input_tensor.size()
        hidden_states = [cell.init_hidden(batch_size, (height, width)) for cell in self.convlstm_cells]

        for t in range(seq_len):
            x = input_tensor[:, t, :, :, :]
            for i, cell in enumerate(self.convlstm_cells):
                h, c = hidden_states[i]
                h, c = cell(x, (h, c))
                hidden_states[i] = (h, c)
                x = h

        output = self.output_layer(x)
        return output

    def predict(self, input_tensor, future_steps):
        self.eval()
        with torch.no_grad():
            current_input = input_tensor
            predictions = []
            for _ in range(future_steps):
                output = self(current_input)
                predictions.append(output)
                current_input = torch.cat([current_input[:, 1:], output.unsqueeze(1)], dim=1)
        return torch.cat(predictions, dim=1)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs, config):
    model.train()
    os.makedirs(config.models_dir, exist_ok=True)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(config.device)
            
            # Use mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, inputs[:, -1, :, :, :])

            # Gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # Save model after each epoch
        model_path = os.path.join(config.models_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

        # Evaluate model after each epoch
        with torch.no_grad():
            predictions = predict_future(model, train_loader, config.future_prediction_start, config.future_prediction_end)
            evaluate_predictions(predictions, train_loader, config.future_prediction_start, config.future_prediction_end)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

    # DataLoader
    train_dataset = LandUseTimeSeriesDataset(config.data_dir, config.sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    
    if config.future_prediction_end > len(train_dataset) + config.sequence_length - 1:
        raise ValueError(f"Future prediction end ({config.future_prediction_end}) is beyond the available data range ({len(train_dataset) + config.sequence_length - 1})")

    # Model initialization
    model = ConvLSTM(input_dim=config.input_dim, hidden_dim=config.hidden_dim, kernel_size=config.kernel_size, num_layers=config.num_layers, output_dim=config.input_dim).to(config.device)

    # Loss function and optimizer
    criterion = config.loss_function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=config.num_epochs, config=config)