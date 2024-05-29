import streamlit as st
import pandas as pd
import nltk
from rake_nltk import Rake
from sentence_transformers import util
import torch
from googletrans import Translator
from transformers import AutoTokenizer, AutoModel
nltk.download('stopwords')
nltk.download('punkt')

def extract_keywords(description):
    print(description)
    translator = Translator()
    rake_nltk_var = Rake()
    translator.raise_Exception = True
    valor = description[:5000].replace('\n'," ")
    text_EN = translator.translate(text=str(valor), dest='en')
    rake_nltk_var.extract_keywords_from_text(text_EN.text)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()

    return " ".join(keyword_extracted)

# count and raking positive and negative keywords
def count_keywords(dataframe,keywords,column:str,keyword_column):
        keyword_values = []
        dataframe[keyword_column] = 0
        for index, row in dataframe.iterrows():
            value = len([i for i in keywords if i in str(row[column]).split(",")])
            keyword_values.append(value)
        dataframe[keyword_column] = keyword_values
        return dataframe

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@st.cache_resource(show_spinner=False)
def load_model(st):
    with st.spinner('Carregando espere um pouco...'):
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return tokenizer, model


# title and description
st.write("""
# Qual é o seu desafio?
""")

# search bar

query = st.text_input("Parece mágica, mas é ciência! ;)", "")

tokenizer, model = load_model(st)

if len(query) > 0:
    query = extract_keywords(query)
    encoded_input_query = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output_query = model(**encoded_input_query)

    query_embeddings = mean_pooling(model_output_query, encoded_input_query['attention_mask'])
    sentence_embeddings = pd.read_pickle('df_tensors.pkl')
    sentence_embeddings = torch.Tensor(list(sentence_embeddings.values))
    lst_cosine_scores = []

    for i in range(len(sentence_embeddings)):
        cosine_scores = util.cos_sim(query_embeddings[0], sentence_embeddings[i])
        lst_cosine_scores.append(cosine_scores)

    lst_cosine_scores_values = []
    for i in range(len(lst_cosine_scores)):
        lst_cosine_scores_values.append(lst_cosine_scores[i].item())

    df_apollo = pd.read_parquet('apollo_base.gzip')

    df_apollo["Cosine score"] = lst_cosine_scores_values
    
    df_cosine_score = df_apollo.rename(columns={"new_description":"text"})
    df_cosine_score.drop_duplicates(inplace=True)
    df_recommendation = df_cosine_score.sort_values(by="Cosine score", ascending=False)
    df_recommendation = df_recommendation.drop(['id_enterprise','country_name','text','enterprise_active','keywords','word_count'], axis=1)
    df_recommendation = df_recommendation[['name', 'website', 'description','Cosine score']]
    df_recommendation = df_recommendation.rename(columns={'name': 'Empresa','description':'Descrição'})
    st.dataframe(df_recommendation.sort_values(by=['Cosine score'], ascending=False).head(n=15))