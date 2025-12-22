# Alzheimer's Stage Detection via MRI

This project uses Deep Learning to classify different stages of Alzheimer's disease based on MRI scan images.

## The structure

*   **`notebook.ipynb`**: The test file that use the trained model to test its accuracy.
*   **`cv-mri.ipynb`**: The main training file that can be reconfigured for better training.
*   **`model.keras`**: The trained and saved Deep Learning model. You can load this directly to make predictions without retraining.
*   **`app.py`**: Web application by streamlit used to demo the model. Used by uploaded the image directly.
*   **`dataset/`**: Contains the MRI images used for training and testing.

## How to install

1.  Clone the repository:
    ```bash
    git clone https://github.com/kei-kochiya/CV-MRI-Alzheimer
    cd CV-MRI-Alzheimer
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If there happens to be some errors along the line of no modules found, please pip install it as I may have forgotten to add it to the requirement files)*

## Running

### 1. Web App
```bash
streamlit run app.py
```
After running this, open the URL provided in the terminal in your browser.

### 2. Retraining or Viewing the Code
```bash
jupyter notebook notebook.ipynb
```
Or just open it in VS Code or anything code editor that support Jupyter Notebook

## Results
*   **Model Accuracy:** ~88%
*   **Model Architecture:** ResNet50

***
