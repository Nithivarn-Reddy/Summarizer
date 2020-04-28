# cs5293sp20-project2

The main aim of this project is to develop a summarizer which takes the data of COVID related literature from Kaggle and summarizes them. By doing so it helps the researchers and the student community to get a summary of all the similar documents which helps them in there respective research work.

High-level overview.
1) Takes a subset of documents of COVID related data (nearly 6000 files out of 60000).
2) Tokenizes and clusters the documents.
3) Finally summarizes each cluster of documents into a file.

The step-wise functionality of achieving the summarization.
1) Explore the data set and look at the format of the files.
2) Function to choose a set of documents in the data set, randomly choose 10 percent of the total files.
3) A file reader a file and tokenize the data.
4) Function to take tokenized data and add it to the clustering method of your choice.
5) Function to take clusters of documents, and summarize the documents.
6) Write the summarized clusters to a file.

### Author - Nithivarn Reddy Shanigaram 

### Email - nithivarn.reddy.shanigaram-1@ou.edu

## External Packages used 

> nltk

> glob

> yellowbrick

> pandas

> numpy

> scikit-learn

> networkx

## Steps to install the project

Open your Terminal..

1) git clone the project 

2) Install pipenv in your system 

  - if it is debain based
    Run the following command
    
    pip install pipenv
    
3) cd into cs5293sp20-project2

4) Run pipenv install
  
 This command installs all the dependencies required by the project. (provided in the Pipfile)

## Steps to Run the project

Now run the project using the following command (Inside the cs5293p20-project2)

 > pipenv shell
  
 > python3 summarizer.py --input "/newdisk/project2-data/" --input "some-value"
  
The Output files is generated inside a Output directory of the current working directory.

## Assumptions made in the project are

1) The data which is downloaded from kaggle is in the form of json. Inside each json file we have several keys having some kind of text as value. After analyzing certain json files and looking at the data, I have only extracted the text data with "body_text" as key. I have also extracted the "paper_id" which is unique for each paper.

2) I am also skipping those papers which don't have any body_text. 

3) For determining the number of clusters , I have used Knee-Elbow method by providing the clusters range of (2,20). I have taken the specified range based on trail and error. My assumption over here is that we can never determine what could be the optimal cluster range as it all depends on the data. For this project since we are choosing 5000 files randomly we cannot determine how close or how far they may be contextually related. 

4) Computationally its taking too much time so I have run the my summarization for one percent of data and have added the output of these files. The instance with 10% of data is also still running.

### Functionality of summarizer.py

This python file is the driving file for the entire project. Here I have registered the input flag to capture its value which are then passed on to the respective methods of project1.py file. I have used argparse for reading the commandline    arguments.

### Functionality of each method in project2.py

#### read_files(path,percent=10):
This method reads the files of the pattern provided by the --input flags and it randomly chooses 10 percent of the files     which account to around 6000 files.This method then returns the file paths of the selected files. I have used glob.glob     function to get the files path. This achieves the step two.
    
#### read_json(data):
This a helper function which takes the json data of each paper and return a dictionary containing only the 'paper_id'       and 'body_text' as keys.

#### json_2_df_2_tokenizer(files):
This is the method which is used for achieving step 3 of this project. Tokenizing the files.I have first clubed all the dictionaries which have been returned by the read_json on each file into a final dictionary.Then I have converted it into a dataframe which now contains 'paper_id' and 'body_text' as columns.Now I extract the 'body_text' column completely as list of strings and pass it to the normalize_corpus function which normalizes the each text and return a list of strings which is the final normalized corpus upon which we need to apply the vectorization. I have used CountVectorizer to generate count vectors of each row(i.e., each 'body_text' string) and I have also experimented with Tfidfvectorizer with various parameters and finally I have settled for Tfidfvectorizer and also set the min_df to 1. I have used Tfidfvectorizer over CountVectorizer because of the number of features and also we need to take into account the inverse document frequency of each word as it helps in clustering the similar documents.This method returns the dataframe and doc_matrix which are generated.So that they can be used by the KMeans and finally to append labels to the clusters.

#### determining_n_clusters(doc_matrix):
For determining the number of optimal clusters for clustering the documents. I have used the knee-elbow method to           determine the optimal k-value. The have taken a KMeans model with max_iter=1000,random_state=42(as mentioned by the         sklearn documentation so that the each time the cluster inits are taken the same),n_jobs='-1' (to enable the parallel       processing of the calculating the clusters). I am then passing this model to the KElbowVisualizer(yellowbrick-package)       with k-value search space in between (2,30). I have taken this specific range based on the computation time and some         trail & error.I have discussed more about this in assumption 3. I have tried to include the visualization so that the       optimal k value is displayed. But since we are running it in a terminal, visualization is not possible so I am using the
KElbowVisualizer's attribute (elbow.value_) to get the optimal cluster value. This function finally returns that 'k'         value.

#### clustering_documents(n_clusters,doc_matrix,df):
The k value determined by the knee-elbow method is passed to the clustering documents method along with the doc_matrix and dataframe returned by the step 3(Tokenization.). Here we fit the KMeans model with the doc_matrix to get the cluster labels for each file. I have used the same parameters as above(max_iter=1000,random_state=42,n_jobs='-1') while initializing the model for KMeans. Finally I have appended the cluster labels to the existing dataframe. Then I have grouped all the documents based on cluster label. I have constructed a dictionary with the cluster_label as key and appended all the documents under one cluster into a single document as its value, so that finally the dictionary contains one document per one cluster.
This dictionary is passed to the next function to summarize the cluster of documents, which is an important step of this project.

#### summarize_clusters(dic_clusters_docs):
This method takes the dictionary with cluster_indexs and documents which is returned by the above method. Over here, I am looping through each item in the dictionary sending the document of each cluster to a summarize() function. Then I am appending the result of the summarize() function along with the respective cluster_index as a tuple into a list.
Finally I am looping through the list which is generated above and writing the summarized text into a SUMMARY-{index}.MD file. So, finally I get SUMMARY-{index}.MD for each cluster which contains the top-10 sentences of each cluster.
This function does both the step 5,6 with the help of helper functions like summarize(), which in turn takes help of several functions which I will be discussing next.

#### summarize(doc,n_top=10):
This is the main method where the actual summarization of the document is done.First I call the parse_document() method to get the list of sentences , then I pass the list of sentences to normalize_corpus() method to get back list of normalized sentences. Then I pass this list of sentences to a Tfidfvectorizer(min_df=0) to generate a dt_matrix.I have used the min_df=0 and left everything to default as it was giving me good results and since normalization is done correctly. Then I am generating the similarity_matrix by multiply dt_matrix with its transpose. This similarity_matrix is used to generate a similarity_graph which is in turn passed to the pagerank algorithm to generate the rank for each sentence. It gives out the rank of each sentence which is then sorted in descending order based on the rank and only the top 10 sentences indices are retrieved. Using the indices retrieved , I am pulling out the actual sentences from the sentences which are returned by the parse_document() method. So finally I am returning a list containing the top 10 sentences.

Reference used - Textbook.

#### parse_document(document):
This method takes each document and checks whether the document content is of string type or not and then converts it into sentences.If the document content is not a string and it is of type unicode , then it converts the unicode content into string and then breaks into sentences. If the document content type is not string or unicode then an Error is raised.
Finally this method returns list of sentences. This is a helper function for summarize(document) method.
Reference - Textbook.

####  normalize_corpus(corpus, lemmatize=True):
This function takes the list of sentences and then tries to normalize the sentence and append it back to a list.I have covered patterns for replacing links, non alphanumeric characters,numeric with space. 
I have also removed the stopwords of the english language along with some other stop words which I have observed while going through the document. I have also lemmatized the words in the sentence so that when we do vectorization we get only the root words. This is the same method which is used while normalizing the data in step 3. It is also reused during creating the summarized text before applying vectorization.

#### remove_stopwords(text):
This method is a helper method for normalize_corpus() which removes the english stop words along with a list of custom added stopwords.

    
#### tokenize_text(text)
This method is a helper method for remove_stopwords() which tokenizes the sentences.
    
#### lemmatize_text(text):
This method takes a text and then using WordNetLemmatizer() lemmatizes the tokens in the text. First we call pos_tag_text() method to get the parts of speech tagging of the each word. Then we pass it to wnl.lemmatize() to lemmatize the word. Finally append all the words back to generate the text.

Reference - Text Book

#### pos_tag_text(text):
This method is a helper function for lemmatize_text() just used for parts of speech tagging of the text. It in turn calls the penn_to_wn_tags()  to generate the tags specific to wnl.lemmatize() method.

Reference - Text Book
    
#### penn_to_wn_tags(pos_tag):
This method just converts the penn tags into wordnet specific tags.

Reference - Text Book

#### Output

Each cluster has a SUMMARY-{index}.MD generated. ( Example SUMMARY-0.MD for cluster 0)
I am then combining all the SUMMARY-{index}.MD into a single SUMMARY.MD file and also adding my approach to it.
