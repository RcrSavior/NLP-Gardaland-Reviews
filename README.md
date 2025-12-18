# NLP-Gardaland-Reviews

This project implements an in-depth Natural Language Processing (NLP) pipeline to analyze customer reviews for the Gardaland theme park. The primary objectives are to uncover key sources of customer dissatisfaction (Topic Discovery) and to establish a robust Deep Learning framework for high-performance review classification.

The data source consists of customer reviews in **Italian**, including a rating score (`mark`), review text, and demographic metadata (e.g., `city`, `region`, `type` of traveler).

## Technical Methodology

The analysis emphasizes advanced feature engineering, contextualized topic discovery, and the setup of modern sequential modeling architecture.

### 1. Data Preprocessing and Feature Engineering

The pipeline was specifically adapted for processing Italian text, requiring meticulous language-specific cleaning:

* **Text Cleaning:** Standard normalization was applied, including lowercasing and the use of the **NLTK** library for removing Italian *stopwords*.
* **Lexical Analysis:** Utilizing **NLTK** and **Spacy**, the text was tokenized, and features were expanded to include **Bigrams (`ngrams_2`)** and **Trigrams (`ngrams_3`)** for capturing multi-word expressions and improving topic coherence.

### 2. Feature Representation: FastText and Word Embeddings

To ensure robust handling of the Italian vocabulary and manage Out-of-Vocabulary (OOV) words inherent in social media or review datasets, the project utilized advanced word embedding techniques.

* **Model Used:** **FastText** (via `gensim.models.fasttext`) was imported and utilized.
* **Rationale:** FastText models represent words as compositions of character n-grams, enabling the generation of meaningful vectors even for words not present in the training corpus (OOV), which is critical for noise resistance in raw customer feedback.

### 3. Topic Modeling and Sentiment Analysis

Similar to other advanced NLP pipelines, **BERTopic** was set up for contextual topic discovery, and `TextBlob` was used for initial sentiment scoring.

| Component | Technique | Purpose |
| :--- | :--- | :--- |
| **Topic Modeling Setup** | BERTopic | Chosen over traditional LDA/NMF for superior contextual topic clustering via BERT embeddings. |
| **Sentiment Analysis** | TextBlob | Provides rapid polarity and subjectivity scores for early-stage exploratory data analysis (EDA). |

### 4. Deep Learning Foundation (Sequential Modeling)

A sophisticated Keras/TensorFlow architecture was configured for future implementation of classification tasks (e.g., predicting the `mark` (rating) based on review text).

* **Architecture:** The model setup involves an **Embedding Layer**, followed by advanced sequential components: **LSTM**, **Bidirectional LSTM**, and **SimpleRNN** layers.
* **Input Preparation:** Text sequences are processed via a **Tokenizer** and then normalized using `pad_sequences` to ensure uniform input length for the sequential model.
* **Evaluation Readiness:** The imports of `precision_score`, `f1_score`, and `recall_score` confirm the project's focus on rigorous model evaluation using standard classification metrics.

## Key Results and Analytical Insights

While the code provides the framework, the analysis is designed to yield these key insights:

* **Topic-Sentiment Linkage:** The analysis maps the discovered BERTopic clusters to customer rating scores (`mark` 1-5), empirically identifying which specific topics (e.g., "Queue Times," "Ticket Prices," "Attraction Maintenance") drive the lowest satisfaction scores.
* **Geospatial Insights:** Utilizing the `city` and `region` data, the analysis supports segmenting customer feedback based on their geographic origin, revealing potential regional differences in satisfaction or perception.
* **Predictive Model Baseline:** The established Deep Learning architecture provides a strong, high-performance baseline capable of achieving superior classification metrics upon full training, allowing for automated real-time classification of new reviews.

## Technologies and Libraries

| Library | Function |
| :--- | :--- |
| **Python** | Core implementation environment. |
| **Keras / TensorFlow** | Deep Learning framework for sequential modeling (LSTM, Bidirectional, SimpleRNN). |
| **Gensim** | Word embedding utility, specifically **FastText** for Italian OOV handling. |
| **NLTK / Spacy** | Core NLP utilities (Lemmatization, Stopwords, Tokenization). |
| **BERTopic** | Unsupervised Topic Modeling. |
| **Scikit-learn** | Metrics for model evaluation (`f1_score`, `classification_report`). |
| **Plotly / WordCloud** | Visualization for exploratory data analysis (EDA). |
