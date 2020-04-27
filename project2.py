import glob
import json
from json import loads,dumps,load
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.cluster import KElbowVisualizer
import networkx
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.cluster import KMeans
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('stopwords')

def read_files(path,percent=10):
    files=glob.glob(path+'/**/**/**/*.json')
    print(len(files))
    files_sample=random.sample(files,int((float(percent)/100)*len(files)))
    return files_sample

def read_json(file):
    d = {}
    dic = loads(open(file).read())
    d["paper_id"]=dic["paper_id"]
    d["body_text"]=[]
    for item in dic["body_text"]:
        d["body_text"].append(item['text'])
    d["body_text"] = ''.join(d["body_text"])
    return d


def json_2_df_2_tokenizer(files):
    final_dict = {'paper_id':[],'body_text':[]}
    for file in files:
        d = read_json(file)
        final_dict['paper_id'].append(d['paper_id'])
        final_dict['body_text'].append(d['body_text'])
    df = pd.DataFrame.from_dict(final_dict)
    print(df)
    #indexes with empty bodytext
    indexes =df[df['body_text']==''].index
    #drop them
    df.drop(indexes,inplace=True)
    #print(df)
    #normalize_function = np.vectorize(normalize_document)
    normalized_corpus = normalize_document(list(df['body_text']))
    #print(normalized_corpus)
    cv = CountVectorizer()
    doc_matrix =cv.fit_transform(normalized_corpus)
    return doc_matrix,df

#using the yellow brick ELbowkneeVisulaizer
def determining_n_clusters(doc_matrix):
    model = KMeans()
    visualizer=KElbowVisualizer(model,k=(2,30))
    visualizer.fit(doc_matrix)
    #visualizer.show()
    #get the optimal cluster value
    k = visualizer.elbow_value_
    return k

def clustering_documents(n_clusters,doc_matrix,df):
    km = KMeans(n_clusters=n_clusters,max_iter=1000,random_state=42,n_jobs=-1)
    km.fit(doc_matrix)
    print(km.labels_)
    df['KMeans_label']=list(km.labels_)
    print(df.info())
    clustered_docs = (df.sort_values(by=['KMeans_label']).groupby(['KMeans_label']))
    df_clustered=clustered_docs.apply(lambda x:x)
    print(df_clustered.columns)
    # converting all clustered rows into dictionary
    dic_clusters_docs = {}
    for i in range(n_clusters):
        doc = []
        print(i)
        for d in df_clustered[df_clustered["KMeans_label"] == i]['body_text']:
            doc.append(d)
        print(len(doc))
        doc_string = ''.join(doc)
        #print(doc_string)
        dic_clusters_docs[i]=doc_string
    return dic_clusters_docs


def summarize_clusters(dic_clusters_docs):
    list_of_summaries=[]
    for (cluster_index,item) in dic_clusters_docs.items():
        list_of_summaries.append((cluster_index,summarize(item)))
    for cluster_index,summary in list_of_summaries:
        if not os.path.exists(os.getcwd()+"/"+"output"):
            os.mkdir("output")
        filename="SUMMARY-{0}".format(cluster_index)+".MD"
        path = os.getcwd()+"/output/"+filename
        with open(path,"w") as f:
            f.write(''.join(summary))

#used for normalizing the text in dataframe
def normalize_document(text,lemmatize=True):
    l = []
    for txt in text:
        txt=re.sub(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+','',txt)
        txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt,flags=re.I|re.A)
        txt = re.sub(r'[0-9]','',txt)
        txt = txt.lower()
        txt = txt.strip()
        txt = remove_stopwords(txt)
        if lemmatize:
            txt = lemmatize_text(txt)
        else:
            txt = txt
        l.append(txt)
    return l


def remove_stopwords(text):
    stopword_list = nltk.corpus.stopwords.words('english')
    other_stop_words = ['report','docs','document']
    stopword_list+=other_stop_words
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def pos_tag_text(text):
    text = nltk.word_tokenize(text)
    tagged_text = nltk.pos_tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text

def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None

def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    wnl = WordNetLemmatizer()
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


def normalize_corpus(corpus, lemmatize=True, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text=re.sub(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+','',text)
        text = text.lower()
        text = remove_stopwords(text)
        #print(type(text))
        #print(text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text,flags=re.I|re.A)
        text = re.sub(r'[0-9]','',text)
        if lemmatize:
            text = lemmatize_text(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
    return normalized_corpus


def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii','ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    #print(sentences)
    return sentences


def summarize(doc,n_top=10):
    sentences = parse_document(doc)
    print(len(sentences))
    norm_sentences = normalize_corpus(sentences)
    vectorizer = TfidfVectorizer(min_df=0)
    dt_matrix = vectorizer.fit_transform(norm_sentences)
    similarity_matrix = (dt_matrix*dt_matrix.T)
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    #networkx.draw_networkx(similarity_graph)
    scores = networkx.pagerank(similarity_graph)
    sentences_rank = sorted(((score,index) for index,score in scores.items()),reverse=True)
    #print(sentences_rank)
    top_indices =[sentences_rank[index][1] for index in range(n_top)]
    print(top_indices)
    summary=[sentences[index] for index in top_indices]
    #print(summary)
    return summary



