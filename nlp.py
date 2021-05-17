import time
import pandas as pd
import glob
import nltk
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer
from tqdm import tqdm
from gensim.models.phrases import Phrases
from gensim.models import Word2Vec
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread

cpus = mp.cpu_count()

nltk.download('punkt')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def remove_hashtags(tokens):
    tokens = map(lambda x: x.replace('#', ''), tokens)
    return list(tokens)

def remove_url(tokens):
    tokens = filter(lambda x: "http" not in x, tokens)
    return list(tokens)

def remove_html(tokens):
    tokens = filter(lambda x: x[0]+x[-1] != '<>', tokens)
    return list(tokens)

def tokenize_url_hashtags(corpus, tweets=False):
    if tweets:
        tokenizer = TweetTokenizer()
    else:
        tokenizer = TreebankWordTokenizer()  
    # Life hack : treebank word tokenizer won't keep html code in one token.
    # To preprocess economics news corpus, use tweettokenizer. 
    tokenized_sentences = []
    for sample in tqdm(corpus):
    # separating sentences
        for sentence in sent_detector.tokenize(sample):
            tokens = tokenizer.tokenize(sentence)
            tokens = remove_url(tokens)
            tokens = remove_html(tokens)
            tokens = remove_hashtags(tokens)
            tokens = list(map(lambda x: x.lower(), tokens))
            tokenized_sentences.append(tokens)
    return tokenized_sentences
    
def nlp_processing(file):
    data = pd.read_csv(file, encoding='latin-1')
    if 'movies_metadata' in file:
        cleaned_data = tokenize_url_hashtags(data.overview.dropna().array)
        mo = Word2Vec(cleaned_data, size=100, window=5, min_count=3)
        mo.train(cleaned_data, total_examples=len(cleaned_data), epochs=10)
    elif 'Political-media' in file:
        cleaned_data = tokenize_url_hashtags(data.text.array, tweets=True)
        pol = Word2Vec(cleaned_data, size=100, window=5, min_count=3)
        pol.train(cleaned_data, total_examples=len(cleaned_data), epochs=10)
    else:
        cleaned_data = tokenize_url_hashtags(data.text.array, tweets=False)
        eco = Word2Vec(cleaned_data, size=100, window=5, min_count=3)
        eco.train(cleaned_data, total_examples=len(cleaned_data), epochs=10)
    
def main():
    files = glob.glob("Data/nlp/*.csv")

    start = time.time()
    for i in files:
        print(i)
        nlp_processing(i)
    end = time.time()
    print("Series computation: {} secs\n".format(end - start))
    
    start = time.time()
    threads = []
    for i in files:
        t = Thread(target=nlp_processing, args=(i,))
        threads.append(t)
        t.start()
      
    for t in threads: t.join()
    end = time.time()
    print("Multithreading computation: {} secs\n".format(end - start))


    start = time.time()
    with mp.Pool(cpus) as p:
        p.map(nlp_processing, files)
        p.close()
        p.join()
    end = time.time()
    print("Multiprocessing computation (with Pool): {} secs\n".format(end - start))

    start = time.time()
    processes = []
    for i in files:
        p = Process(target=nlp_processing, args=(i,))
        processes.append(p)
        p.start()
      
    for p in processes: p.join()
    end = time.time()
    print("Multiprocessing computation (with Process): {} secs\n".format(end - start))

if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    main()