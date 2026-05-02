# Explainable Aspect-Based Sentiment Analysis using Transformer Models 

## (Transformer and Not Transformer Models for Aspect-Based Sentiment Analysis (ABSA))

This repository contains the code and datasets used for fine-tuning transformer models and training Bi-LSTM for Aspect-Based Sentiment Analysis (ABSA). It also includes various explainability methods to interpret model predictions, such as LIME, SHAP, Integrated Gradients, and Grad-CAM.

### Introduction

This project focuses on applying transformer-based models (e.g., BERT, RoBERTa, XLNet) and Bi-LSTM to the task of Aspect-Based Sentiment Analysis (ABSA), allowing us to predict the sentiment associated with specific aspects in text. The repository includes preprocessing code, model fine-tuning and training scripts, and several explainability methods to understand model decisions.

The datasets used in this project include a combination of SemEval, MAMS, and Hu and Liu customer reviews, along with an additional test dataset from Naver Labs Europe.

### Block Diagram of the Framework
![image](https://github.com/user-attachments/assets/d506a80a-e33d-4392-9fb5-4748a8edafd3)


The above diagram illustrates the overall workflow of our Aspect-Based Sentiment Analysis (ABSA) frame-work. The process begins with data collection, where reviews containing labeled aspects and their corresponding sentiment polarities are gathered. Then we create a labeled dataset and fine-tune pre-trained transformer models (such as BERT and RoBERTa) on this data. After fine-tuning, we evaluate the models using performance metrics (e.g., accuracy, F1 score) and select the best-performing model. We used also some basic Not Transformer models like LSTM, Bi-LSTM and RNN for baseline models to see the comparison between them and the fine-tuned pre-trained transformer models. Finally, explainability techniques (LIME, SHAP, Attention Weights Visualization, Integrated Gradients, Grad-CAM) are applied to the selected model to provide insights into how the model makes predictions at the aspect level. This diagram provides a visual guide to the methodological steps outlined in this section, from data preparation to explainability analysis.



### File Descriptions

1. **FINAL_CLEANED_CORRECTED_SHUFFLED_DATASET_NO_DUPLICATE.jsonl**

This is the main dataset used in the project after preprocessing, which includes removing duplicates, null aspects, and combining the SemEval, MAMS, and Hu_and_Liu_customer_reviews datasets. It is used for the fine-tuning of transformer models and for training and evaluating the Bi-LSTM.

2. **NAVER_DATASET.jsonl**

This dataset is structured similarly to the main dataset and contains reviews from Naver Labs Europe. It is used to test the generalization ability of the models on unseen data from a different domain.

3. **FINE_TUNE.py**

This script fine-tunes transformer models for ABSA.

Key configurations include:

1. Separators: Ensure the appropriate separator is used based on the model:
- `[SEP]`: for BERT, ALBERT, DistilBERT, Bi-LSTM
- `</s></s>`: for RoBERTa
- `<sep>`: for XLNet
2. Hyperparameters: You can adjust key hyperparameters:
- Learning Rate: `lr` in the optimizer
- Batch Size: `batch_size` in the dataloaders
- Epochs: `epochs`
3. Model Selection: Change the model and tokenizer by modifying these lines:

```
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
```
Replace 'distilbert-base-uncased' with your desired model, e.g., 'roberta-base', but ensure the correct separator is used.

4. **TRAIN_BI-LSTM.py**

This script trains the Bi-LSTM model. Key configurations include:

Hyperparameters: Adjust the learning rate (`lr`), batch size (`batch_size`), and number of epochs (`n_epochs`).
The Bi-LSTM model uses the `[SEP]` separator, which has been added to its vocabulary.

5. **TRAIN_LSTM.py**

This script trains the LSTM model. Key configurations include:

Hyperparameters: Adjust the learning rate (`lr`), batch size (`batch_size`), and number of epochs (`n_epochs`).
The LSTM model uses the `[SEP]` separator, which has been added to its vocabulary.

6. **TRAIN_RNN.py**

This script trains the RNN model. Key configurations include:

Hyperparameters: Adjust the learning rate (`lr`), batch size (`batch_size`), and number of epochs (`n_epochs`).
The RNN model uses the `[SEP]` separator, which has been added to its vocabulary.

7. **LIME.py**

This script implements the LIME (Local Interpretable Model-agnostic Explanations) method for explainability. You can load your fine-tuned transformer model using the `model_dir` or `model_path`, define the `tokenizer`, and test the explainability on custom input text and aspects. Make sure to use the appropriate separator for your model.

8. **SHAP.py**

This script implements SHAP (SHapley Additive exPlanations) to interpret the transformer models. Load the model and tokenizer, set the correct separator, and input the text and aspects for which you want to explain the sentiment predictions.

9. **ATTENTION_WEIGHTS.py**

This script calculates and visualizes attention weights to interpret model predictions. Similar to the other explainability scripts, load the model, tokenizer, and text input, ensuring the correct separator is used.

10. **INTEGRATED_GRADIENTS.py**

This script applies the Integrated Gradients method for explaining the predictions of transformer models. After loading your fine-tuned model and tokenizer, and specifying the appropriate separator, you can input your text and aspect for analysis.

11. **GRAD-CAM.py**

This script implements GRAD-CAM (Gradient-weighted Class Activation Mapping) to visualize how specific parts of the text contribute to the model’s predictions. Make sure to load the transformer model, tokenizer, and set the correct separator.

### Running Your Own Experiments

To run your own experiments, follow these steps:

- Prepare the Dataset: Use either the provided `FINAL_CLEANED_CORRECTED_SHUFFLED_DATASET_NO_DUPLICATE.jsonl` or another dataset structured similarly (with text and aspect labels).

- Choose Your Model: In `FINE_TUNE.py`, select the model you wish to fine-tune by setting the `AutoTokenizer` and `AutoModelForSequenceClassification`. For the Bi-LSTM model, use the `TRAIN_BI-LSTM.py` script. For the LSTM model, use the `TRAIN_LSTM.py` script. For the RNN model, use the `TRAIN_RNN.py` script.

- Adjust Hyperparameters: Modify the learning rate, batch size, and number of epochs in the respective scripts to suit your experiment.

- Run Explainability Methods: Once your models are trained, use the explainability scripts (e.g., LIME, SHAP, Integrated Gradients) to interpret the model's predictions. Ensure the model paths and tokenizers are correctly set in each script.

Conclusion

This repository provides all the necessary tools for fine-tuning transformer models and training Bi-LSTM, LSTM, RNN for ABSA, along with methods to interpret model decisions. Feel free to experiment with different models, datasets, and hyperparameters, and leverage the explainability techniques to gain deeper insights into the models' decision-making processes.
