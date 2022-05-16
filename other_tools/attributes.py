from scipy.io import loadmat
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
f = os.path.join(os.path.dirname(__file__),'..','data','attrann.mat')
x = loadmat(f)

x = x['attrann']
a = [d[0] for d in x['attributes'][0][0][0]]
l = x['labels'][0][0]
b = x['bboxes'][0][0]
i = x['images'][0][0]

pos = (l > 0)* 1
co_pos = pos.T.dot(pos)
co_pos < 4
neg = (l < 0)*1
co_neg = pos.T.dot(neg)
infrequent1 = [(a[x],a[y]) for x,y in zip(*np.where(co_pos == 0)) if x < y]
t = np.quantile(co_neg, 0.95)
infrequent2 = [(a[x],a[y]) for x,y in zip(*np.where(co_neg >  t)) if x < y]

def meta_redirect(content):
    soup  = BeautifulSoup(content.decode('ascii'))

    result=soup.find("meta",attrs={"http-equiv":"refresh"})
    if result:
        wait,text=result["content"].split(";")
        if text.strip().lower().startswith("url="):
            url=text[4:]
            return url
    return None

def get_content(url):
    response = requests.get(url)
    content = response.content
    # follow the chain of redirects
    while meta_redirect(content):
        resp = requests.get(meta_redirect(content))
    return resp.content

def get_wordnet(wnid):
    resp = requests.get( f'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={wnid}')
    return resp.content.decode('ascii').replace('\n', ';')

def get_labels():
    labels = {}
    with open(os.path.join(os.path.dirname(__file__),'..','data','labels.txt'),'r') as fp:
        lines = fp.readlines()
        for line in lines:
            mnid = line.split(':')[0]
            label = line.split(':')[1]
            labels[mnid] = label
    return labels

#with open('labels.txt','a') as fp:
#    for x in range(i.shape[0]):
#        wnid = i[x][0][0].split('_')[0]
#        if wnid not in labels:
#            words = get_wordnet(wnid)
#            fp.write(f'{wnid}:{words}\n')
#            labels[wnid] = words
