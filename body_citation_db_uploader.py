# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:28:55 2020

@author: niran
"""

## Making of secondary Mongo DB which has citations count about the content of research paper referenced with cord_uid
from pymongo import MongoClient
import json
import pandas as pd

file_name = 'metadata.csv'
metadata_df = pd.read_csv(file_name, low_memory=False)
main_frame_df = pd.DataFrame(metadata_df, columns = ['cord_uid', 'title', 'abstract', 'pdf_json_files', 'url'])


#paper_df = pd.DataFrame()
total_citations = 0
total_covid_citations = 0
br = 0
citation_count = {}

client = MongoClient()
db = client["thesis_database"]
table = db.paper_citation

# Looping through the body text of reasearch papers
for i in range(100000):
       
    if i % 10000 == 0 :
        print("On" + str(i) + " Total Citations: " + str(total_citations) + " Total Covid Citations: " + str(total_covid_citations))
        br = br + 1
        #if br == 2:
        #    break
    paper_loc = main_frame_df.iloc[i]['pdf_json_files']
    cord_uid = main_frame_df.iloc[i]['cord_uid']
    
   
   
    #print(type(paper_loc))
    if isinstance(paper_loc , str):
        try:
            paper_loc = paper_loc.split(';')
            for current_paper_loc in paper_loc:
                current_paper_loc = current_paper_loc.strip()
                with open(current_paper_loc) as paper_json:
                    paper_data = json.load(paper_json)
                    
                    biblio_dict = paper_data['bib_entries']
                    
                    for body_text in paper_data['body_text'] :
                        references = []
                        
                        cite_spans = body_text['cite_spans']
                        #print("CITE:SPANS", cite_spans)
                        if cite_spans is not None:
                            for cite_span in cite_spans: 
                                #print("cite ", cite_span['ref_id'], "bib", biblio_dict)
                                if cite_span['ref_id'] is not None:
                                    references.append(biblio_dict[cite_span['ref_id']]['title'])
                        current_row = {}
                        current_row['cord_uid'] = cord_uid
                        current_row['body_text'] = body_text['text']
                        current_row['cite_spans'] = cite_spans
                        current_row['references'] = references
                        
                        #print(current_row)
                        table.insert(current_row)
                    """
                    #Making the citation count distribution
                    for citation in biblio_dict:
                        if biblio_dict[citation]['title'] in citation_count.keys():
                            citation_count[biblio_dict[citation]['title']] = citation_count[biblio_dict[citation]['title']] + 1
                        else :
                            citation_count[biblio_dict[citation]['title']] = 1
                    
                    # Adding all references to the total count
                    len_biblio = len(paper_data['bib_entries'])
                    total_citations = total_citations + len_biblio                    
                    
                    title = main_frame_df.iloc[i]['title']
                    if isinstance(title,str):
                        if title.find('covid') or title.find('-cov-2') or title.find('cov2') or title.find('ncov'):
                            total_covid_citations = total_covid_citations + 1 
                            #print("len(paper_data['bib_entries']) " + str(len(paper_data['bib_entries'])) + " Total covid" + str(total_covid_citations))
    #print("len(paper_data['bib_entries']) " + len(paper_data['bib_entries']) + " Total " + str(total_citations) + " Total cov:" +str(total_covid_citations))"""             
        except FileNotFoundError:
            print( " File not found in " + paper_loc)
