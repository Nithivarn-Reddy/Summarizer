# cs5293sp20-project2

The main aim of this project is to develop a summarizer which takes the data of COVID related literature from Kaggle and summarizes them. By doing so it helps the researchers and the student community to get a summary of all the similar documents which helps them in there respective research work.

1) Takes a subset of documents of COVID related data (nearly 6000 files out of 60000).
2) Tokenizes and clusters the documents.
3) Finally summarizes each cluster of documents into a file.


### Author - Nithivarn Reddy Shanigaram 

### Email - nithivarn.reddy.shanigaram-1@ou.edu

### Structure
```
.
└── cs5293p20-project-1
    ├── COLLABORATORS
    ├── LICENSE
    ├── Pipfile
    ├── Pipfile.lock
    ├── README.md
    ├── files
    │   ├── modi.redacted.txt
    │   └── text1.redacted.txt
    ├── modi.txt
    ├── otherfiles
    │   ├── test.md
    │   └── test_1.txt
    ├── project1
    │   ├── __init__.py
    │   ├── project1.py
    │   └── redactor.py
    ├── setup.py
    ├── tests
    │   ├── __init__.py
    │   └── test_redact.py
    └── text1.txt
```

External Packages used 

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
  
This will display the result in your console / terminal.


## Assumptions made in the project are

1) The data which is downloaded from kaggle is in the form of json. Inside each json file we have several keys having some kind of text as value. After analyzing certain json files and looking at the data, I have only extracted the text data with "body_text" as key. I have also extracted the "paper_id" which is unique for each paper.

2) I am also skipping those papers which don't have any body_text. 

3) For determining the number of clusters , I have used Knee-Elbow method by providing the clusters range of (2,20). I have taken the specified range based on trail and error. My assumption over here is that we can never determine what could be the optimal cluster range as it all depends on the data. For this project since we are choosing 5000 files randomly we cannot determine how close or how far they may be contextually related. 

4) Computationally its taking too much time so I have run the my summarization for one percent of data and have added the output of these files. The instance with 10% of data is also still running.

### Functionality of summarizer.py

    This python file is the driving file for the entire project. Here I have registered the input flag to capture its value which are then passed on to the respective methods of project1.py file. I have used argparse for reading the commandline     arguments.

### Functionality of each method in project2.py

#### read_files(path,percent=10):
    This method reads the files of the pattern provided by the --input flags and it randomly chooses 10 percent of the files     which account to around 6000 files.This method then returns the file paths of the selected files. I have used glob.glob     function to get the files path.
    
#### read_json(data):
    This a helper function which takes the json data of each paper and return a dictionary containing only the 'paper_id'       and 'body_text' as keys.

#### json_2_df_2_tokenizer(files):
    This is the method which is used for achieving task 2 of this project. Tokenizing the files.
    I have first clubed all the dictionaries which have been returned by the read_json on each file into a final dictionary.
    Then I have converted it into a dataframe which now contains 'paper_id' and 'body_text' as columns.
    Now I extract the 'body_text' column completely as list of strings and pass it to the normalize_document function which     normalizes the each text and return a list of strings which is the final normalized corpus upon which we need to apply       the vectorization. I have used CountVectorizer to generate count vectors of each row(i.e., each 'body_text' string) and     I have also experimented with Tfidfvectorizer with various parameters and finally I have settled for Tfidfvectorizer and     also set the min_df to 1. I have used Tfidfvectorizer over CountVectorizer because of the number of features and also we     need to take into account the inverse document frequency of each word as it helps in clustering the similar documents. 
    This method returns the dataframe and doc_matrix which are generated.So that they can be used by the KMeans and finally     to append labels to the clusters.

#### determining_n_clusters(doc_matrix):
    For determining the number of optimal clusters for clustering the documents. I have used the knee-elbow method to           determine the optimal k-value. The have taken a KMeans model with max_iter=1000,random_state=42(as mentioned by the         sklearn documentation so that the each time the cluster inits are taken the same),n_jobs='-1' (to enable the parallel       processing of the calculating the clusters). I am then passing this model to the KElbowVisualizer(yellowbrick-package)       with k-value search space in between (2,30). I have taken this specific range based on the computation time and some         trail & error.I have discussed more about this in assumption 3. I have tried to include the visualization so that the       optimal k value is displayed. But since we are running it in a terminal, visualization is not possible so I am using the
    KElbowVisualizer's attribute (elbow.value_) to get the optimal cluster value. This function finally returns that 'k'         value.
#### redact_concept(data,concept):
    Approach for extracting concept :
    I am passing data to into the redact_concept function along with the concept to be searched for. I have used wordnet         synset derived from nltk.corpus for getting all the synsets related to the concept. Then I have searched for hypernymns,     homonymns,holonymns of each synset of the concept. I have not taken into account the meronymns as they talk about part       of the concept or the substances which contain the concept. I have formed a list of words from each synset of the           concept and searched whether a sentence contains any word of the list of words(the                                           synonymns,hypernymns,hyonymns,holonymns). If the sentence contains any word then it is redacted. This approach may have     certain errors as wordnet works well for verbs and noun.I am appending all the redacted sentences to a global list so       that the stats function can use it to display the statistics.

Reference - Text book.

#### redact_stats(args):
    This method takes a string which is passed as an argument. It generates a dictionary of consisting of                       names,genders,dates,redacted_sentences,concept_words as keys and lists of counts of names , genders, dates which are         generated by the above methods and the redacted_sentences contain the list of all sentences which have some connection       with the concept. I am also additionally adding a key containing the concept_words which are generated around a concept.
    Finally if args is a either stdout or stderr then it is written to the standardout and standarderror files respectively.
    If args is other than stdout,stderr then a txt file with the name provided in the argument is created and the dictionary     is written as string.
    
#### write_output(data,outpath):
    This method takes as input the final data which is the redacted data and also the value of the --output flag. It then       checks if a directory mentioned in the outpath is present. If not we create a new directory and write all our .redacted     appended files into that directory.
    
#### Testcases
    
#### test_redact.py
    
    This python file contains the test definitions which I have used.
    
#### test_readFile(pattern):
    This method checks whether the actual readFiles method of the project1.py is returning correct data when I pass certain     patterns.
    
#### test_redact_names(sampledata):
    This method is testing the functionality of the redact_names of project1.py. I am passing in a List of String containing     just the person names and checking whether the method is returning any data and it is correctly redacting the names or       not.
    
#### test_redact_dates(sampledata):
     This method is testing the functionality of the redact_dates of project1.py. I am passing in a List of String                containing just the date formats and checking whether the method is returning any data and it is correctly redacting        the dates provided in the sample data.
     
#### test_redact_genders(sampledata):
      This method is testing the functionality of the redact_genders of project1.py. I am passing in a List of String             containing just the gender names and checking whether the method is returning any data and it is correctly redacting         the genders.
      
#### test_redact_concept(sampledata,concept):
       This method is testing the functionality of the redact_concept of project1.py. I am passing in a List of String              containing a sentence and the concept which the function it needs to search for. I am checking whether the method            is returning any data and len(redacted_sentences) is not zero.
      
#### test_redact_stats(capsys):
        I am testing whether stats is printing out its data to the correct stream or not, by calling stats('stdout') and checking if an empty string is printed in the stdout.
