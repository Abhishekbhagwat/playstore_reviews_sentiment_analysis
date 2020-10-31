# Play Store Sentiment Analysis
![Hugging Face](https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/transformers_logo_name.png)

This app is built using Hugging Face Transformers in the backend for Sentiment Analysis using NLP techniques. It uses Streamlit in the frontend to predict sentiments on raw text.

### Installation Guide

1. Clone the repo onto your local system using
```git
git clone https://github.com/Abhishekbhagwat/playstore_reviews_sentiment_analysis.git
```
2. Install all the requirements in the `requirements.txt` file in your newly created virtual environment
3. Once this is done, you can either train your own model from scratch by running the model training again by using
```python
python3 model/bert_model.py
```
4. Or alternatively you can make use of the best pre-trained model from us. This can be downloaded by using
```bash
bash assets/download_model.sh
```
5. Once this is done, you can run the streamlit app by
```
streamlit run streamlit_app.py
```

If you prefer to use a REST API, we also have that which is made from `fastapi`.
