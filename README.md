

# Toxic Comment Classification - using BERT

This project uses a **BERT-based** machine learning model to identify and classify toxic comments from Reddit. The model is trained on the **train.csv** dataset and evaluated on **test.csv**, focusing on various types of toxicity such as "toxic," "severe\_toxic," "obscene," etc and compare its performance against two classical baselines:

* TF–IDF + Logistic Regression
* TF–IDF + Calibrated SVM



## Dataset

* **Train Dataset**: `train.csv` - Contains comments labeled with various toxicity types.
* **Test Dataset**: `test.csv` - Used for evaluating the model.

[Link to Dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/)

## Features

* **Data Preprocessing**: Handles missing data and performs tokenization.
* **Feature Selection**: Utilizes BERT embeddings for feature extraction.
* **Modeling**: Implements a BERT-based model for classifying comments.
* **Evaluation**: Model performance is evaluated using **F1 score**, **Accuracy**, and **ROC Curves**.
* **Visualization**: Generates visualizations like **WordCloud** for toxic comments and **Clustering** of comment features.

## Requirements

* **Python 3.x**
* **Libraries**:

  * `transformers`
  * `torch`
  * `scikit-learn`
  * `matplotlib`
  * `seaborn`
  * `datasets`
  * `wordcloud`
  * `tqdm`
  * `pandas`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## How to Use

### Clone the Repository

Clone the repository with the following command:

```bash
git clone https://github.com/denise08/255Project.git
cd 255Project.git
```

### Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Prepare the Dataset

Upload the **train.csv** and **test.csv** files to your environment. The dataset files should be in the root directory.

1. **Running in Kaggle**:
   We ran this project in **Kaggle**, where the tokenizer is specified as:

   ```python
   tokenizer = BertTokenizer.from_pretrained("/kaggle/input/bert-base-uncased/bert-base-uncased")
   ```

2. **Running in Google Colab**:
   If you're running it on **Google Colab**, you may need to change the runtime from **CPU** to **GPU**:

   * Go to **Runtime > Change runtime type**.
   * Set **Hardware Accelerator** to **GPU**.

   Then, replace the tokenizer line in the script with:

   ```python
   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
   ```

3. Alternatively, you can run the `.py` file directly:

```bash
python ./script.py
```

### Expected Outputs

1. **Model Evaluation Metrics**:

   * **F1 Score**, **Accuracy**, and other metrics after training and evaluation.

2. **ROC Curves**:

   * Visual representations for each label (e.g., toxic, severe\_toxic).

3. **F1 Score and Accuracy vs Epochs**:

   * Plots showing the change in **accuracy** and **F1 score** over each epoch during training.

4. **WordCloud**:

   * Visualize the most frequent words in toxic comments using a word cloud.

5. **Clustering**:

   * KMeans clustering of BERT embeddings to group similar toxic comments.

## Project Report

You can view the detailed project report [here](https://docs.google.com/document/d/1nJ86w01LInrdJ7eJFH7MWkX0hvRUzneJ1xPbpKxGWlY/edit?tab=t.0).





