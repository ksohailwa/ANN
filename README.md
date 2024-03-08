# ANN
# Artificial Neural Network for Power Plant Energy Output Prediction

This repository contains code for building and training an Artificial Neural Network (ANN) using TensorFlow for predicting the energy output of a power plant based on certain features.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

Make sure you have the following installed on your system:

- Python (version 3.6 or later)
- Jupyter Notebook (optional)
- TensorFlow (version 2.2.0-rc2)
- NumPy
- pandas

You can install TensorFlow and other dependencies using pip:

```bash
pip install tensorflow==2.2.0-rc2 numpy pandas
```

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/your_repository.git
```

2. Navigate to the project directory:

```bash
cd your_repository
```

3. Download the dataset `Folds5x2_pp.xlsx` and place it in the project directory.

### Usage

Follow the instructions in the Jupyter Notebook or run the Python script to preprocess the data, build the ANN model, train the model, and make predictions.

## Dataset

The dataset used in this project is `Folds5x2_pp.xlsx`, which contains features and the corresponding energy output of a combined cycle power plant.

## Model Architecture

The ANN model architecture consists of an input layer, two hidden layers with ReLU activation functions, and an output layer.

- Input Layer: Number of neurons = Number of features in the dataset
- Hidden Layers: Number of neurons = 6, Activation function = ReLU
- Output Layer: Number of neurons = 1 (for predicting energy output)

## Training

The model is trained using the Adam optimizer and mean squared error loss function. It is trained for 100 epochs with a batch size of 32.

## Results

After training the model, predictions are made on the test set, and the results are printed, showing the predicted energy output alongside the actual energy output.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is inspired by [OpenAI](https://openai.com) and the vast community contributing to advancements in machine learning and artificial intelligence.
