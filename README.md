# 📱 P719: NLP Sentiment Analysis on Amazon Phone Reviews

A machine learning project that reads a customer review of a phone and predicts whether the sentiment is **Positive**, **Neutral** or **Negative**. The best model is deployed as a web app with Streamlit.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]

### 🔴 Live App: [https://sentiment-analysis-dkoa46fknf4fcmnk7gzzan.streamlit.app/)

> Note: on Streamlit Cloud's free tier, the app goes to sleep after a few days without visitors. If you see a "wake up" button, click it and wait a minute.

---

## 📌 Table of Contents
1. [Business Objective](#-business-objective)
2. [Dataset](#-dataset)
3. [Project Workflow](#-project-workflow)
4. [Results](#-results)
5. [Why Logistic Regression Was Chosen](#-why-logistic-regression-was-chosen)
6. [The Web App](#-the-web-app)
7. [Project Structure](#-project-structure)
8. [How to Run Locally](#-how-to-run-locally)
9. [How to Deploy on Streamlit Cloud](#-how-to-deploy-on-streamlit-cloud)
10. [Tech Stack](#-tech-stack)
11. [Limitations and Future Work](#-limitations-and-future-work)
12. [Team](#-team)

---

## 🎯 Business Objective
Extract the sentiment (Positive / Neutral / Negative) from customer reviews of a product. This helps a business understand what customers feel without reading thousands of reviews by hand.

## 📊 Dataset
- **File:** `P719-Dataset.xlsx`
- **Size:** 1,440 Amazon phone reviews, no missing values and no duplicate rows
- **Columns:** `title`, `rating` (1 to 5 stars) and `body` (the review text)

The dataset only has star ratings, so the target column `sentiment` is created from the rating:

| Rating | Sentiment | Reviews |
|---|---|---|
| 4 and 5 stars | Positive | 729 |
| 3 stars | Neutral | 199 |
| 1 and 2 stars | Negative | 512 |

The classes are imbalanced: Neutral is the smallest class, which makes it the hardest one to predict.

## 🔄 Project Workflow
The full work is in the notebook `P719_G_1_Sentiment_Analysis.ipynb`.

1. **Import libraries**
2. **Load data** from the Excel file
3. **EDA:** shape, missing values, duplicates, rating distribution, creating the sentiment target
4. **Text cleaning:** join title and body, convert to lowercase, keep only letters, remove extra spaces
5. **Feature engineering:** character count, word count, TextBlob polarity and subjectivity
6. **Visualization:** review length, most common words, polarity charts, and a TextBlob-only baseline
7. **Data preparation:** 80/20 train-test split (stratified, `random_state=42`), TF-IDF (3,000 features, single words and word pairs), and TextBlob polarity added as one extra feature
8. **Model building:** 5 models trained and evaluated one by one
9. **Final evaluation:** comparison table and chart, best model selected by F1-score
10. **Conclusion**
11. **Deployment** with Streamlit

## 🏆 Results
The TextBlob-only baseline reached **67.64%** accuracy. All five machine learning models did better. Results on the test set (288 reviews):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.747 | 0.741 | 0.747 | **0.744** |
| SVM (Linear) | 0.753 | 0.728 | 0.753 | 0.737 |
| Decision Tree | 0.743 | 0.725 | 0.743 | 0.731 |
| Random Forest | 0.778 | 0.741 | 0.778 | 0.726 |
| Naive Bayes | 0.760 | 0.664 | 0.760 | 0.702 |

Precision, recall and F1-score are weighted averages over the three classes.

**Logistic Regression, class by class:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 0.78 | 0.80 | 0.79 |
| Neutral | 0.30 | 0.28 | 0.29 |
| Positive | 0.84 | 0.84 | 0.84 |

## ✅ Why Logistic Regression Was Chosen
- It has the **highest weighted F1-score** (0.744) of all five models.
- It is the only model that handles all three classes reasonably. Random Forest and Naive Bayes have higher accuracy, but they almost never predict Neutral (Neutral recall of 0.03 and 0.00), so their accuracy comes mostly from the two bigger classes.
- It gives **probabilities** (for example "82% Positive"), so the app can show how confident the model is. Linear SVM cannot do this.
- It is simple, fast and small, which suits Streamlit Cloud.

## 🖥️ The Web App
👉 **Try it here:** [PASTE_YOUR_APP_LINK_HERE](PASTE_YOUR_APP_LINK_HERE)

The app (`app.py`) trains the model once when it starts, using exactly the same steps as the notebook, and keeps it in memory so it stays fast.

**Features**
- **Single review tab:** type or paste a review, or click an example button, and get the predicted sentiment in a coloured card with the confidence for each class
- **Many reviews tab:** upload a CSV or Excel file, pick the column with the reviews, see a summary chart and download the results as a CSV file
- **Sidebar:** shows the model's test accuracy, weighted F1-score and number of reviews used
- A note appears when the model is not very confident (top probability below 50%)

## 📁 Project Structure
```
├── app.py                                  # Streamlit web app
├── requirements.txt                        # Libraries needed by the app
├── P719-Dataset.xlsx                       # Dataset (the app trains on it at startup)
├── P719_G_1_Sentiment_Analysis.ipynb       # Full notebook: EDA, models, evaluation, deployment code
└── README.md                               # This file
```

## 💻 How to Run Locally
1. Download or clone this repository.
2. Open a terminal in the project folder and install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```
4. The app opens in your browser (usually at `http://localhost:8501`).

To explore the analysis, open `P719_G_1_Sentiment_Analysis.ipynb` in Jupyter Notebook. In the notebook the dataset path is a local path (`C:\Users\...`), so change it to where your copy of the file is.

## ☁️ How to Deploy on Streamlit Cloud
1. Put `app.py`, `requirements.txt` and `P719-Dataset.xlsx` in a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose the repository, set the main file to `app.py` and click **Deploy**.
4. After a minute or two the app is live and you get a public link.

The dataset file name must be exactly `P719-Dataset.xlsx`, because `app.py` reads that name.

## 🛠️ Tech Stack
- **Language:** Python
- **Data:** pandas, numpy, openpyxl
- **Visualization:** matplotlib
- **NLP:** TextBlob, TF-IDF (scikit-learn)
- **Machine learning:** scikit-learn (Logistic Regression, Naive Bayes, Linear SVM, Decision Tree, Random Forest), scipy
- **Deployment:** Streamlit, Streamlit Cloud, GitHub

## ⚠️ Limitations and Future Work
**Limitations**
- **Neutral reviews are the weakest class** (F1 of about 0.29 for Logistic Regression). They have the fewest examples, and they often mix good and bad opinions. For example, the notebook's test review "It is okay, nothing special for the price" was predicted Positive.
- The dataset is small (1,440 reviews) and only covers phone reviews in English, so the model may not work well on other products.
- Sentiment labels come from star ratings, which do not always match the text of the review (for example sarcasm).
- The metrics shown are from a single 80/20 split.

**Possible improvements**
- Collect more data, especially Neutral reviews
- Try cross-validation and hyperparameter tuning
- Try deep learning models such as BERT-based models
- Add more analysis in the app, such as the most important words behind each prediction

## 👥 Team
P719, Group 1

*Add the team members' names here.*
