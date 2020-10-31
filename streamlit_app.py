import streamlit as st
from sentiment_classifier.model import Model, get_model
import re
import spacy_streamlit
import en_core_web_sm
nlp = en_core_web_sm.load()

def clean_text(text):
# remove all unicode characters. In this case = emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text

st.title('Play Store Sentiment Analysis')
st.markdown('''
Welcome! You can analyse the sentiment of the reviews of the apps found on Google Play Store. 
This app makes use of BERT Transformers by HuggingFace in the backend to predict the sentiment of the review. 
You may also find the inbuilt tokenizer handy :)
 ''')


st.subheader('Review Classification')
review_input = st.text_input('Enter your review here:')
review_input = clean_text(review_input)

st.subheader("Available Tasks")
docx = nlp(review_input)
if st.button("Tokenize Text"):
    spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_'])
model = get_model()

if st.button("Predict Sentiment"):
    if review_input!='':
        with st.spinner('Predicting...'):
            sentiment, confidence, probabilities = model.predict(review_input)
            st.markdown('__Sentiment__ of this review is: `{0}`.'.format(sentiment))
            st.markdown('The confidence probabilities of the classes are `{0}`'.format(probabilities))