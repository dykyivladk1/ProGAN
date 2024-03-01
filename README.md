# ProGAN Implementation

This repository contains the implementation of Progressive Growing of GANs. This type of GAN is designed to generate high-quality images.

## Getting Started

Follow these steps to use this implementation:

### Prerequisites

Ensure you have Python installed on your system. This code is compatible with Python 3.9 and newer versions.

### Dataset

For training and testing the ProGAN model, you'll need a dataset. I used CelebA dataset which you download using the following link:

[CelebA Link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

After downloading, place the dataset in an appropriate directory within the your project structure, such as "./data".

### Installation

1. **Clone the repository** to your local computer:

    ```
    git clone https://github.com/dykyivladk1/ProGAN.git
    ```


2. **Install the required dependencies**. It's recommended to create and use a virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Training model**

    To train a model for custom dataset, you can use the following command:
    
    ```
    python scripts/trainer.py --train_dir <train_path>
    ```
4. **Visualisations**

    You can use Netron app for opening the **.onnx** files stored in visualisations folder. I used them for understanding the model structure.

5. **Note**
   
   If you want to see my documentation for this model you can visit the following link on Notion:

   [Documentation](https://dykyivladk.notion.site/ProgressiveGAN-cff025d80a794ba7811ad632b36c9600?pvs=4)
