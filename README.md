# Machine Learning Project

This project is designed to train a machine learning model using tweet data. The model will be saved in HDF5 format for future use.

## Project Structure

```
ml-project
├── REIMPLY
│   └── tweets.csv          # Training data in CSV format
|   └── model_1.h5            # Trained model
│   └── predict_gui.py     # GUI for predicting sentiment
|   └── requirements.txt   # Python dependencies
|   └── tokenizer.pickle   # Tokenizer for text  preprocessing
|   └── train.py           # Training script
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone https://github.com/TLxGHOST/Sentinetal-anlysis
   cd ml-project
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Running the Training Script

To train the model, run the following command:

```
python train.py
```

This will load the data from `tweets.csv`, preprocess it, train the model, and save the trained model as `model_1.h5`.

## Additional Information

- Ensure that you have the necessary permissions to use the data in `tweets.csv`.
- Modify the training parameters in `train.py` as needed to improve model performance.
