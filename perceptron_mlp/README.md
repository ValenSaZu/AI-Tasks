## Project Structure

```
PerceptronMLP
├── src
│   ├── main.cu                # Entry point of the application
│   ├── perceptron_mlp.h       # Definition of the Perceptron class
│   ├── cifar10_reader.h       # DataLoader class for CIFAR-10 dataset
│   ├── layers
│   │   ├── dense_layer.h       # Definition of the DenseLayer class
│   │   ├── dense_layer.cu      # Implementation of DenseLayer methods
│   │   ├── activation.h         # Activation function declarations
│   │   └── activation.cu        # Implementation of activation functions
│   ├── losses
│   │   ├── softmax_loss.h      # Definition of the SoftmaxLoss class
│   │   └── softmax_loss.cu     # Implementation of SoftmaxLoss methods
│   ├── utils
│   │   ├── cuda_utils.h        # Utility functions for CUDA
│   │   └── rng_utils.h         # Utility functions for random number generation
│   └── tests
│       ├── test_forward.cpp     # Unit tests for forward pass
│       └── test_backward.cpp    # Unit tests for backward pass
├── scripts
│   └── download_cifar.sh       # Script to download CIFAR-10 dataset
├── CMakeLists.txt              # CMake configuration file
├── Makefile                     # Build instructions for the project
├── .gitignore                   # Files to ignore in Git
├── LICENSE                      # Licensing information
└── README.md                    # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd PerceptronMLP
   ```

2. **Install dependencies**:
   Ensure you have CUDA installed on your machine. Follow the installation instructions for your operating system.

3. **Download the CIFAR-10 dataset**:
   Run the script to download the dataset:
   ```
   cd scripts
   ./download_cifar.sh
   ```

4. **Build the project**:
   You can build the project using either CMake or Makefile.

   Using CMake:
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

   Using Makefile:
   ```
   make
   ```

5. **Run the application**:
   After building, you can run the application:
   ```
   ./main
   ```

## Usage

The application will load the CIFAR-10 dataset, train the MLP model, and evaluate its performance. The training progress will be displayed in the console.

## Acknowledgments

- The CIFAR-10 dataset is provided by the Canadian Institute for Advanced Research (CIFAR).
- This project is inspired by various resources on neural networks and CUDA programming.