import argparse
import project2
import pickle

global final_data
if __name__=="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--input",type=str,required=True,action='append',help="It takes the patterns of the files")
    args=parser.parse_args()
    if args.input:
        if len(args.input) == 2:
            sample_files = project2.read_files(args.input[0],args.input[1])
        else:
            sample_files = project2.read_files(args.input[0])
"""
dc = pickle.load(open("/home/nithivarn_gmail_com/matrix.txt","rb"))
df = pickle.load(open("/home/nithivarn_gmail_com/df.txt","rb"))
"""


if len(sample_files):
    #second function which takes the file paths and returns a doc_matrix and dataframe
    doc_matrix,df=project2.json_2_df_2_tokenizer(sample_files)
    k = project2.determining_n_clusters(doc_matrix)
    print(k)
else:
    print("File path is incorrect")

dic_clusters=project2.clustering_documents(k,doc_matrix,df)
project2.summarize_clusters(dic_clusters)

