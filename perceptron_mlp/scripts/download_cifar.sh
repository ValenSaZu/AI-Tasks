#!/bin/bash

# Script to download the CIFAR-10 dataset

# Define the URL for the CIFAR-10 dataset
CIFAR10_URL=https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

# Download the dataset
echo "Downloading CIFAR-10 dataset..."
curl -O $CIFAR10_URL

# Extract the dataset
echo "Extracting CIFAR-10 dataset..."
tar -xzvf cifar-10-binary.tar.gz

# Clean up the tar file
rm cifar-10-binary.tar.gz

echo "CIFAR-10 dataset downloaded and extracted successfully."