import torch
import torch.nn as nn


class SimpleRULPredictor(nn.Module):
    """
    A simple LSTM + fully connected model for RUL prediction
    """
    def __init__(self, input_size, hidden_size=64, num_layers=8, dropout=0.2):
        """
        Initialize the model
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(SimpleRULPredictor, self).__init__()
        self.name = "SimpleRULPredictor"
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),

        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1) representing RUL
        """
        # LSTM expects input of shape (batch_size, sequence_length, input_size)
        lstm_out, _ = self.rnn(x)
        # lstm_out, _ = self.lstm(x)
        
        # Use only the last output from the LSTM
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(last_output)
        
        return output


class CNNRULPredictor(nn.Module):
    """
    A CNN-based model for RUL prediction
    """
    def __init__(self, input_size, sequence_length, num_filters=64, kernel_size=3, dropout=0.2):
        """
        Initialize the model
        
        Args:
            input_size: Number of input features
            sequence_length: Length of the input sequence
            num_filters: Number of CNN filters
            kernel_size: Size of the CNN kernel
            dropout: Dropout probability
        """
        super(CNNRULPredictor, self).__init__()
        self.name = "CNNRULPredictor"
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate size after convolutions and pooling
        self.flat_size = num_filters * 2 * (sequence_length // 4)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1) representing RUL
        """
        # Reshape for CNN: (batch_size, sequence_length, input_size) -> (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        output = self.fc_layers(x)
        
        return output


class HybridRULPredictor(nn.Module):
    """
    A hybrid CNN+LSTM model for RUL prediction
    """
    def __init__(self, input_size, sequence_length, cnn_filters=32, lstm_hidden=64, dropout=0.2):
        """
        Initialize the model
        
        Args:
            input_size: Number of input features
            sequence_length: Length of the input sequence
            cnn_filters: Number of CNN filters
            lstm_hidden: Size of hidden layers in LSTM
            dropout: Dropout probability
        """
        super(HybridRULPredictor, self).__init__()
        self.name = "HybridRULPredictor"
        
        # CNN for feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.rnn = nn.GRU(input_size=cnn_filters, hidden_size=lstm_hidden, num_layers=2, batch_first=True,
                          dropout=dropout)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1) representing RUL
        """
        # Reshape for CNN: (batch_size, sequence_length, input_size) -> (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)
        
        # CNN feature extraction
        x = self.conv(x)  # Shape: (batch_size, cnn_filters, sequence_length)
        
        # Reshape for LSTM: (batch_size, cnn_filters, sequence_length) -> (batch_size, sequence_length, cnn_filters)
        x = x.permute(0, 2, 1)
        
        # LSTM processing
        # lstm_out, _ = self.lstm(x)
        gru_out, _ = self.rnn(x)
        
        # Use only the last output from the LSTM
        x = gru_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(x)
        
        return output
