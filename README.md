# cs5293sp20-project2

The main aim of this project is to develop a summarizer which takes the data of COVID related literature from Kaggle and summarizes them. By doing so it helps the researchers and the student community to get a summary of all the similar documents which helps them in there respective research work.

.1) Takes a subset of documents of COVID related data (nearly 5000 files out of 60000).
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

Now run the project using the following command (Inside the cs5293p20-project-1)

 > pipenv run python project1/redactor.py --input '*.txt' \
                    --input 'otherfiles/*.md' \
                    --names --dates \
                    --concept 'kids' \
                    --output 'files/' \
                    --stats stdout
  
This will display the result in your console / terminal.

## To run testcases 
Go inside the virtual environment by running the following commands.

1) cd cs5293p20-project-1

2) Run pipenv shell

3) Run pytest

## Assumptions made in the project are

1) The data which is downloaded from kaggle is in the form of json. Inside each json file we have several keys having some kind of text as value. After analyzing certain json files and looking at the data, I have only extracted the text data with "body_text" as key. I have also extracted the "paper_id" which is unique for each paper.

2) I am also skipping those papers which don't have any body_text. 

3) For determining the number of clusters , I have used Knee-Elbow method by providing the clusters range of (2,30). I have taken the specified range based on trail and error. My assumption over here is that we can never determine what could be the optimal cluster range as it all depends on the data. For this project since we are choosing 5000 files randomly we cannot determine how close or how far they may be contextually related. 

4) 

5) I have taken two text files modi.txt , text1.txt and one markdown file under otherfiles/ for my execution.

### Functionality of redactor.py

    This python file is the driving file for the entire project. Here we register all the flags and capture there values         which are then passed on to the respective methods of project1.py file. I have used argparse for reading the commandline     arguments.

### Functionality of each method in project1.py

#### readFiles(pattern="*.txt"):
    This method reads the files of the pattern provided by the --input flags. 
    Input - ['*.txt','..'] - List of strings 
    It has a lamda function just with a read functionality and this method returns data in the read from the files in the       form list of strings.This list of data is used by the other methods to redact the data.
    
#### redact_names(data):
    Approach for extracting names:
    During extraction of names from the text files , first I have broken all paragraphs into sentences and later each           sentence into individual words,there by,creating a words_list.Then applied the pos_tag to get the parts of speech tag       for each word and then applied ne_chunk to get the named_entities like PERSON names from the text. But this approach was     giving me extra words like GOODS, SERVICE, TEX so on. But later, I realized that the above approach of creating a           words_list and then applying pos_tag is wrong as the parts of speech of each word depends on the context where it is         used in the sentence. So I have changed my approach and applied ne_chunk on pos_tagged words of each sentence and           extracted the named_entities from them. This approach has given me good results.
    For easing through the tree parse I have also used a lambda function.
    Returns : name redacted data list , list of counts of names redacted in each file. A global list for count is maintained     which is used for statistics.
References used: (https://stackoverflow.com/questions/14841997/how-to-navigate-a-nltk-tree-tree)

#### redact_genders(data):
    Approach for extracting genders from text:
    During the extraction of genders from the text , I have tried using WordNet and synoymns but it didn't workout as the       corpus for synoymns and antonymns are to less, so I have explicitly taken a list of male_words and female_words and         combined them into a list of gender_words . I have also added camel-cased gender_words to the gender_words list and then     redacted the words which are part of this list.
    Returns: gender redacted data list , list of counts of genders redacted in each file. The count of genders in each file     is appended to a global list so that it can be used for statistics.
References used: (http://nealcaren.github.io/text-as-data/html/times_gender.html)

#### redact_dates(data):
    Approach for matching dates :
    date formats considered = 'dd/mm/yyyy | dd-mm-yyyy | yyyy-mm-dd | yyyy/mm/dd | yyyy-dd-mm | yyyy/dd/mm | mm/dd/yyyy |       mm-dd-yyyy | dd (January-December|jan-dec|Jan-Dec|january-decemeber) , yyyy | dd(th) (January-December|jan-dec|Jan-         Dec|january-decemeber) yyyy | dd(st) (January-December|jan-dec|Jan-Dec|january-decemeber) yyyy | (January-December|jan-     dec|Jan-Dec|january-decemeber) dd , yyyy .
    I am extracting the dates of the above format and redacting them. For this to work , I have written a regular expression     that matches the above mentioned date formats. It doesn't redact text containing only months and year or only year.
    Returns : dates redacted data list , list of counts of dates redacted in each file. The count of the dates in each file     are appending to a global list so that it can be used for statistics.
References used: (https://stackoverflow.com/questions/10308970/matching-dates-with-regular-expressions-in-python)                            (https://docs.python.org/3/library/re.html)

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
