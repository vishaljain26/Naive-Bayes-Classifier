import csv
import random
import re
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import time
start_time = time.clock()

finaltokens=[]
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
def loadCsv(filename):
	lines = csv.reader(open(filename, "rt",encoding='LATIN-1'))
	dataset=list(lines)
	dataset.pop(0)
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


def bagofwords(file1,file2,file3):
    finalwords=[]
    data1={}
    data2={}
    data3={}
    for x in range(len(file1)):
        data1[file1[x][1]]=file1[x][2]

    for x in range(len(file2)):
        data2.setdefault(file2[x][0],[]).append(file2[x][2])

    for x in range(len(file3)):
        data3[file3[x][0]] = file3[x][1]

        #data3.setdefault(file3[x][0],[]).append(file3[x][1])
    for key in data2:
        data2.setdefault(key,[]).append(data1.get(key))
        data2.setdefault(key, []).append(data3.get(key))


    return data2


file1=loadCsv("C:/Users/visha/Downloads/Documents/New folder/train.csv")
file2=loadCsv("C:/Users/visha/Downloads/Documents/New folder/attributes.csv")
file3=loadCsv("C:/Users/visha/Downloads/Documents/New folder/product_descriptions.csv")
finalwords=[]
total_data=bagofwords(file1,file2,file3)
for key in total_data.keys():
    finalwords.append(total_data.get(key))
finalwords=str(finalwords).lower()


def refinestring(finalwords):

    tokens = tokenizer.tokenize(finalwords)
    cachedstopwords = stopwords.words('english')
    doc = ' '.join([word for word in tokens if word not in cachedstopwords])
    return doc

doc=refinestring(finalwords)
#print(doc)
for word in doc.split(' '):
    if word not in finaltokens:
        finaltokens.append(stemmer.stem(word))
#print(finaltokens)

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

f1=separateByClass(file1)
list1=f1.get('1')
list1=list(list1)
list1.extend(f1.get('1.33'))
list2=f1.get('2')
list2=list(list2)
list2.extend(f1.get('1.67'))
list2.extend(f1.get('2.33'))
list3=f1.get('3')
list3=list(list3)
list3.extend(f1.get('2.67'))


newdic1={}
newdic2={}
newdic3={}
for x in range(len(list1)):
    newdic1[list1[x][1]]=list1[x][2]
for x in range(len(list2)):
    newdic2[list2[x][1]] = list2[x][2]
for x in range(len(list3)):
    newdic3[list3[x][1]]=list3[x][2]
relevance1=[]
relevance2=[]
relevance3=[]
for key in newdic1.keys():
    relevance1.append(total_data.get(key))
for key in newdic2.keys():
    relevance2.append(total_data.get(key))
for key in newdic3.keys():
    relevance3.append(total_data.get(key))

rel1=tokenizer.tokenize(str(relevance1))
rel2=tokenizer.tokenize(str(relevance2))
rel3=tokenizer.tokenize(str(relevance3))

for x in range(len(rel1)):
    rel1[x]=stemmer.stem(rel1[x])
for x in range(len(rel2)):
    rel2[x]=stemmer.stem(rel2[x])
for x in range(len(rel3)):
    rel3[x]=stemmer.stem(rel3[x])


rel1=' '.join(rel1).lower()
rel2=' '.join(rel2).lower()
rel3=' '.join(rel3).lower()

#print(rel1)
#print(rel2)
#print(rel3)
prob1={}
prob2={}
prob3={}
coun=0
for word in finaltokens:

    prob1[word]=(rel1.count(word)+1)/(len(rel1)+len(finaltokens))
    prob2[word] = (rel2.count(word) + 1) / (len(rel2) + len(finaltokens))
    prob3[word] = (rel3.count(word) + 1) / (len(rel3) + len(finaltokens))
#print(lis)
#print(prob1)
#print(prob2)
#print(prob3)
#for key in newdic:
 #   print(tot_data.)
#getquery(fi)


def prediction(search_term):
    total=len(list1)+len(list2)+len(list3)
    p1=len(list1)/total
    p2 = len(list2) / total
    p3 = len(list3) / total


    tokens=tokenizer.tokenize(search_term)
    for x in range(len(tokens)):
        tokens[x]=stemmer.stem(tokens[x])
    temp1=1
    temp2 = 1
    temp3 = 1
    for x in range(len(tokens)):
        if tokens[x] not in prob1.keys():
            temp1=temp1*1
            temp2 = temp2 * 1
            temp3 = temp3 * 1
            continue
        temp1=temp1*float(prob1.get(tokens[x]))
        temp2 = temp2 * float(prob2.get(tokens[x]))
        temp3 = temp3 * float(prob3.get(tokens[x]))
    final_prob1=p1*temp1
    final_prob2=p2*temp2
    final_prob3=p3*temp3

    #print(final_prob1)
   # print(final_prob2)
    #print(final_prob3)
    findmax=max(final_prob1, final_prob2, final_prob3)
    if findmax==final_prob1:
        return 1
    if findmax==final_prob2:
        return 2
    else:
        return 3

file4 = loadCsv("C:/Users/visha/Downloads/Documents/New folder/test.csv")
result={}
for x in range(len(file4)):
    result[file4[x][0]]=prediction(file4[x][3])

writer = csv.writer(open('result.csv', 'wt'), delimiter=',', lineterminator='\n')
writer.writerow(["id","relevance"])
for key, value in result.items():
   writer.writerow([key, value])
print("--- %s seconds ---" % (time.clock() - start_time))

#search_term='lawn sprkinler'
#prediction(search_term)