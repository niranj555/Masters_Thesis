#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import torch
import re
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
import nltk
from nltk.corpus import stopwords
import json
import pickle
import string
from rank_bm25 import BM25Okapi


#Make primary dataframe : cord_uid, title, abstract, pdf_json_files, url

file_name = 'metadata.csv'
metadata_df = pd.read_csv(file_name, low_memory=False)
main_frame_df = pd.DataFrame(metadata_df, columns = ['cord_uid', 'title', 'abstract', 'pdf_json_files', 'url'])


paper_loc = main_frame_df.iloc[0]['pdf_json_files']
with open(paper_loc) as paper_json:
      data = json.load(paper_json)
      print(len(data['bib_entries']))



import math

from rank_bm25 import BM25Okapi

english_stopwords = list(set(stopwords.words('english')))

class CovidGenie:
    """
    Simple CovidGenie.
    """
    
    def remove_special_character(self, text):
        #Remove special characters from text string
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text):
        # tokenize text
        words = nltk.word_tokenize(text)
        return list(set([word for word in words 
                         if len(word) > 1
                         and not word in english_stopwords
                         and not word.isnumeric() 
                        ])
                   )
    
    def preprocess(self, text):
        # Clean and tokenize text input
        return self.tokenize(self.remove_special_character(text.lower()))


    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus
        self.columns = corpus.columns
        
        raw_search_str = self.corpus.abstract.fillna('') + ' '                             + self.corpus.cord_uid.fillna('')
        
        self.index = raw_search_str.apply(self.preprocess).to_frame()
        self.index.columns = ['terms']
        self.index.index = self.corpus.index
        self.bm25 = BM25Okapi(self.index.terms.tolist())
    
    def search(self, query, num):
        """
        Return top `num` results that better match the query
        """
        # obtain scores
        search_terms = self.preprocess(query) 
        doc_scores = self.bm25.get_scores(search_terms)
        
        # sort by scores
        ind = np.argsort(doc_scores)[::-1][:num] 
        
        # select top results and returns
        results = self.corpus.iloc[ind][self.columns]
        results['score'] = doc_scores[ind]
        print("----------------------BM25 SCORES---------------------------")
        print(doc_scores[ind] )
        results = results[results.score > 0]
        return results.reset_index()


"""
Download pre-trained QA model
"""

import torch
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

BERT_SQUAD = 'bert-large-uncased-whole-word-masking-finetuned-squad'

model = BertForQuestionAnswering.from_pretrained(BERT_SQUAD)
tokenizer = BertTokenizer.from_pretrained(BERT_SQUAD)

model = model.to(torch_device)
model.eval()

print()

type(tokenizer)



def answer_question(question, context):
    # answer question given question and context
    encoded_dict = tokenizer.encode_plus(
                        question, context,
                        add_special_tokens = True,
                        max_length = 256,
                        pad_to_max_length = True,
                        return_tensors = 'pt'
                   )
    
    input_ids = encoded_dict['input_ids'].to(torch_device)
    token_type_ids = encoded_dict['token_type_ids'].to(torch_device)
    
    start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    
    answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])
    answer = answer.replace('[CLS]', '')
    return answer


NUM_CONTEXT_FOR_EACH_QUESTION = 10


def get_all_context(query, num_results):
    # Return ^num_results' papers that better match the query
    
    papers_df = cse.search(query, num_results)
    return papers_df['abstract'].str.replace("Abstract", "").tolist(), papers_df['pdf_json_files'].tolist()

def get_all_paper_content_context(question, all_content_json_loc ):
    # Return top context based on the question and paper content
    
    content_context = []
    
    for json_loc in all_content_json_loc:
        if isinstance(json_loc , str):
            json_loc = json_loc.split(';')
            for current_paper_json in json_loc:
                current_paper_json = current_paper_json.strip()
                with open(current_paper_json) as paper_json:
                    paper_data = json.load(paper_json)
                    paper_content = [ content['text'] for content in paper_data['body_text'] ]

                if paper_content is not None:
                    tokenized_content = [content.split(" ") for content in paper_content ]
                    bm25 = BM25Okapi(tokenized_content)
                    tokenized_question = question.split(" ")
                    result = bm25.get_top_n(tokenized_question, paper_content, n=1)

                    content_context.append(result)
    
    return content_context

def get_all_answers(question, all_contexts):
    # Ask the same question to all contexts (all papers)
    
    all_answers = []
    
    for context in all_contexts:
        all_answers.append(answer_question(question, context))
    return all_answers


def create_output_results(question, 
                          all_contexts, 
                          all_answers, 
                          summary_answer='', 
                          summary_context=''):
    # Return results in json format
    
    def find_start_end_index_substring(context, answer):   
        search_re = re.search(re.escape(answer.lower()), context.lower())
        if search_re:
            return search_re.start(), search_re.end()
        else:
            return 0, len(context)
        
    output = {}
    output['question'] = question
    output['summary_answer'] = summary_answer
    output['summary_context'] = summary_context
    results = []
    for c, a in zip(all_contexts, all_answers):

        span = {}
        span['context'] = c
        span['answer'] = a
        span['start_index'], span['end_index'] = find_start_end_index_substring(c,a)

        results.append(span)
    
    output['results'] = results
        
    return output

    
def get_results(question, 
                summarize=False, 
                num_results=NUM_CONTEXT_FOR_EACH_QUESTION,
                verbose=True):
    # Get results

    all_contexts, all_content_json = get_all_context(question, num_results)
    
    print("############# QUESTIONS #########################")
    print(question)
    
    print("######################## ALL CONTEXTS ###############################")
    print(all_contexts)
    
    print("########### ALL JSON TOP CONTENT################")
    print(all_content_json)
    
    all_paper_content_contexts = get_all_paper_content_context(question, all_content_json)
    print( all_paper_content_contexts )
    
    all_answers = get_all_answers(question, all_contexts)
    
    print("########################### ALL ANSWERS #######################################")
    print(all_answers)
    
    if summarize:
        # NotImplementedYet
        summary_answer = get_summary(all_answers)
        summary_context = get_summary(all_contexts)
    
    return create_output_results(question, 
                                 all_contexts, 
                                 all_answers)


covid_kaggle_questions = {
"data":[
          {
              "task": "What is known about transmission, incubation, and environmental stability?",
              "questions": [
                  "Is the Coronavirus / COVID-19 transmitted by aerisol, droplets, food, close contact, fecal matter, or water?",
                  "How long is the incubation period for the Coronavirus / COVID-19?",
                  "Can the Coronavirus / COVID-19 be transmitted asymptomatically or during the incubation period?",
                  "How does weather, heat, and humidity affect the tramsmission of Coronavirus / COVID-19?",
                  "How long can the Coronavirus / COVID-19 remain viable on common surfaces?"
              ]
           },
          {
              "task": "What do we know about COVID-19 risk factors?",
              "questions": [
                  "What risk factors contribute to the severity of 2019-nCoV?",
                  "How does hypertension affect patients?",
                  "How does heart disease affect patients?",
                  "How does copd affect patients?",
                  "How does smoking affect patients?",
                  "How does pregnancy affect patients?",
                  "What is the fatality rate of 2019-nCoV?",
                  "What public health policies prevent or control the spread of 2019-nCoV?"
              ]
          },
          {
              "task": "What do we know about virus genetics, origin, and evolution?",
              "questions": [
                  "Can animals transmit 2019-nCoV?",
                  "What animal did 2019-nCoV come from?",
                  "What real-time genomic tracking tools exist?",
                  "What geographic variations are there in the genome of 2019-nCoV?",
                  "What effors are being done in asia to prevent further outbreaks?"
              ]
          },
          {
              "task": "What do we know about vaccines and therapeutics?",
              "questions": [
                  "What drugs or therapies are being investigated?",
                  "Are anti-inflammatory drugs recommended?"
              ]
          },
          {
              "task": "What do we know about non-pharmaceutical interventions?",
              "questions": [
                  "Which non-pharmaceutical interventions limit tramsission?",
                  "What are most important barriers to compliance?"
              ]
          },
          {
              "task": "What has been published about medical care?",
              "questions": [
                  "How does extracorporeal membrane oxygenation affect 2019-nCoV patients?",
                  "What telemedicine and cybercare methods are most effective?",
                  "How is artificial intelligence being used in real time health delivery?",
                  "What adjunctive or supportive methods can help patients?"
              ]
          },
          {
              "task": "What do we know about diagnostics and surveillance?",
              "questions": [
                  "What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV?"
              ]
          },
          {
              "task": "Other interesting questions",
              "questions": [
                  "What is the immune system response to 2019-nCoV?",
                  "Can personal protective equipment prevent the transmission of 2019-nCoV?",
                  "Can 2019-nCoV infect patients a second time?"
              ]
          }
   ]
}



#paper_df.rename(columns={'paper_ref': 'cord_uid'}, inplace=True)

#combined_df = pd.merge(main_frame_df, paper_df, how='inner', on = 'cord_uid')
#combined_df.head()
cse = CovidGenie(main_frame_df)



main_frame_df.head()



all_tasks = []

for i, t in enumerate(covid_kaggle_questions['data']):
    print("Answering questions to task {}. ...".format(i+1))
    answers_to_question = []
    for q in t['questions']:
            answers_to_question.append(get_results( q, verbose=False))
    task = {}
    task['task'] = t['task']
    task['questions'] = answers_to_question
    print("TASK",task)
    all_tasks.append(task)

