#--- TEXT EXTRACTION ---#

#importing  libraries
import PyPDF2
from nltk.tokenize import sent_tokenize

#reading document
pdf_file = open('JavaBasics-notes.pdf', 'rb')
read_pdf = PyPDF2.PdfFileReader(pdf_file)
number_of_pages = read_pdf.getNumPages()

#extracting text
text = []
for no in range(0,number_of_pages) :
    page = read_pdf.getPage(no)
    page_content = page.extractText()
    page_content = sent_tokenize(page_content)
    #removing footer
    text = str(text)+str(page_content[2:])

#------------------------------#

#--- TEXT CLEANING ---#
    
#importing libraries
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#unessential data  elimination
text = re.sub('[^a-zA-Z]',' ',text)

#tokenizing
text = word_tokenize(text)

#stopword removal
text = [word for word in text if not word in set(stopwords.words('english'))]

#-----------------------------#

#--- KEYWORD EXTRACTION ---#

#importing libraries
from gensim.summarization import keywords
from sklearn.feature_extraction.text import TfidfVectorizer

#keyword determination by textRank algorithm
joined_text = ' '.join(text)
textRank = keywords(joined_text,scores = True)

#keyword determination by tf-idf algorithm
tv = TfidfVectorizer(max_features =100)
tv.fit_transform(text)
tf_idf = tv.get_feature_names()

#-----------------------------#

#--- WEIGHTAGE OF KEYWORDS ---#

#frequency dictionary
textRank = dict(textRank)
l = []
for w in tf_idf:
    if w in textRank:
        l.append((w,textRank[w])) 

#sorting by weightage
def getKey(item):
    return item[1]
l= sorted(l, key=getKey,reverse = True)

#-----------------------------#

#--- WRITING TO FILE ---#
         
#importing libraries
import csv

#writing to file 
with open('./result.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['keyword','weightage'])
    for row in l:
        csv_out.writerow(row)

#-----------------------------#













