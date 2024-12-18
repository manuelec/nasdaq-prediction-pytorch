# nasdaq-prediction

## 1- AutoGluon TS Forecasting - NASDAQ (Jupyter Notebook)

## 2- Advanced Time Series Forecasting with PyTorch - NASDAQ (Jupyter Notebook)
The project implements an advanced time series forecasting model using PyTorch to predict NASDAQ leveraging covariates like S&P 500, VIX stock indices.
The Jupyter Notebook *Advanced_Time_Series_PyTorch_V2.ipynb* was developed in Google Colab using CPU.
Following you can find an high level description of the workflow implementation, but my advice is to look at the Notebook worflow that includes brief descriptions for esch step.

### Dataset
The dataset (yfinance) includes the following features:
  
- NASDAQ closing prices (target)
- S&P 500 closing prices
- 10-Year Treasury Yield (DGS10)
- CBOE Volatility Index (VIX)

Additional engineered features:
- Log-transformed prices
- Returns
- Volatility
- Moving averages
- Relative strength
- Data is preprocessed to handle missing values and create derived features.

### Model Architecture
- The TimeSeriesAttention model is a neural network designed for time series forecasting that incorporates an attention mechanism. .
```python
class TimeSeriesAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super(TimeSeriesAttention, self).__init__()
        self.hidden_size = hidden_size

        # Attention components
        self.attention = nn.Linear(hidden_size, 1)

        # Main network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Project to hidden space
        h = F.relu(self.bn1(self.fc1(x.view(batch_size, -1))))

        # Apply attention
        attention_weights = F.softmax(self.attention(h), dim=1)
        attended = h * attention_weights

        # Final prediction
        out = self.dropout(attended)
        return self.fc2(out)
```

### Key Features
* Attention Mechanism: Allows the model to focus on different parts of the input sequence.
* Batch Normalization: Helps stabilize training and potentially improves generalization.
* Dropout: Reduces overfitting by randomly zeroing some neurons during training.


### Training
The train_and_validate function implements a comprehensive training and validation process for the time series forecasting model. Here are the key features:

* Gradient Accumulation: The function uses gradient accumulation (set to 2 steps) to effectively increase the batch size without increasing memory usage.
* Learning Rate Warmup: A warmup period of 3 epochs is implemented, gradually increasing the learning rate to improve training stability.
* Dynamic Learning Rate: Utilizes ReduceLROnPlateau scheduler to adjust the learning rate based on validation performance.
* Early Stopping: Implements early stopping with a patience of 5 epochs to prevent overfitting.
* Gradient Clipping: Applies gradient clipping with a max norm of 1.0 to prevent exploding gradients.
* Best Model Saving: Saves the best model state based on validation loss.
* NaN Checking: Checks for NaN values in each batch to ensure training stability.
* Gradient Norm Monitoring: Prints the gradient norm every 10 batches for monitoring purposes.

The function iterates through the specified number of epochs, performing the following steps in each epoch:

- Training Phase:
 - - Forwards data through the model
 - - Calculates loss and backpropagates
 - - Accumulates gradients and updates weights
 - - Applies gradient clipping
- Validation Phase:
 - - Evaluates the model on the validation set
 - - Calculates validation loss
- Model Saving and Early Stopping:
 - - Saves the best model based on validation loss
 - - Triggers early stopping if no improvement for a set number of epochs
- Learning Rate Adjustment:
 - - Adjusts learning rate based on validation performance

### Evaluation
Model performance is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R2) score

### Grid Search and Cross Validation
Hyperparameter Tuning
The model utilizes Grid Search with Cross Validation for hyperparameter optimization:
- Grid Search: Systematically works through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance.
- Cross Validation: Uses k-fold cross-validation to assess model performance more robustly.
- Hyperparameters Tuned:
- - Number of LSTM layers
- - Number of neurons in each layer
- - Dropout rate
- - Learning rate
- - Batch size
- Best Parameters: The optimal hyperparameters found through this process are:
- - LSTM layers: 2
- - Neurons per layer: 64
- - Dropout rate: 0.2
- - Learning rate: 0.001
- - Batch size: 32

### Visualization
The notebook includes visualizations of:
- Feature correlation matrix
- Training and validation loss curves
- Predicted vs actual values

### Requirements
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

### Usage
- Simply clone the Jupyter Notebook
- Install the required dependencies
- Run the Jupyter Notebook to train the model and make predictions, make any change you desire
