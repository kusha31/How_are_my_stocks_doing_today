#!/usr/bin/env python
# coding: utf-8

# # 1. Import some libraries

# In[1]:


get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install transformers')


# In[2]:


# Import dependencies from transformers huggingface module
# financial-summarization-pegasus model trained by huggingface
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Beautiful soup will help to scrape data from news reports
from bs4 import BeautifulSoup

# Need requests to request data from the web, which then can be processed using beautiful soup
# Data: all stock news articles from maybe yahoo finance, CNBC, etc.,
import requests

import re
from transformers import pipeline
import csv


# # 2. Setup a summarization model

# In[3]:


# Use the financial-summarization-pegasus already model trained from huggingface 
model_name = "human-centered-summarization/financial-summarization-pegasus"

# Creating word-embeddings from the input text --> every single word is represented by a unique identifier/vector
tokenizer = PegasusTokenizer.from_pretrained(model_name)

model = PegasusForConditionalGeneration.from_pretrained(model_name)


# In[12]:


# GEt all data from the url (paragraphs from the article)
url = "https://www.fool.com/investing/2022/03/08/how-berkshire-hathaway-could-use-its-144-billion/?source=eptyholnk0000202&utm_source=yahoo-host&utm_medium=feed&utm_campaign=article"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
paragraphs = soup.find_all('p') # looking for all the text using 'p' tag


# In[13]:


paragraphs


# In[14]:


r


# In[15]:


r.text


# In[16]:


paragraphs[0].text


# In[17]:


text = [paragraph.text for paragraph in paragraphs]
words = ' '.join(text).split(' ')[:400] # grab first 400 words from each paragraph.
ARTICLE = ' '.join(words) # join all the words to make an article


# In[18]:


words


# In[19]:


len(words)


# In[21]:


ARTICLE


# In[22]:


input_ids = tokenizer.encode(ARTICLE, return_tensors='pt') # encode the article
output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True) # run the model, it uses bean search algorithm basically
summary = tokenizer.decode(output[0], skip_special_tokens=True) # decode the output


# In[23]:


# Print out the summary of the article
summary


# # 4. Building an news and sentiment pipeline

# In[26]:


# Here I'm trying to get the news articles for Moderna, Apple and Ethereum stocks
monitored_tickers = ['MRNA', 'APPL', 'ETH']


# ## 4.1 Search for stock news using google and yahoo finance

# In[27]:


def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a') # 'a' tags look for links to articles
    hrefs = [link['href'] for link in atags]
    return hrefs


# In[ ]:


search_for_stock_news_urls('APPL')


# In[ ]:


raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitored_tickers}
raw_urls


# In[33]:


raw_urls.keys()


# In[34]:


raw_urls.values()


# ## 4.2 Strip out unwanted URLs

# In[36]:


import re # regular expressions library
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

# The URLs should have https:// and no words from the exclude_list
def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val)) # remove any duplicate URLs


# In[37]:


strip_unwanted_urls(raw_urls['MRNA'], exclude_list)


# In[38]:


cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}
cleaned_urls


# ## 4.3 Search and scrape cleaned URLs

# In[50]:


def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs: 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES


# In[51]:


articles = {
    ticker:scrape_and_process(cleaned_urls[ticker]) 
            for ticker in monitored_tickers
}
articles


# In[52]:


articles['ETH'][3]


# ## 4.4 Summarize all articles

# In[53]:


def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


# In[54]:


summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}
summaries


# In[58]:


summaries['MRNA']


# # 5. Adding sentiment analysis

# In[59]:


from transformers import pipeline
sentiment = pipeline('sentiment-analysis')


# In[60]:


sentiment(summaries['MRNA'])


# In[61]:


scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}
scores


# In[62]:


print(summaries['ETH'][3], scores['ETH'][3]['label'], scores['ETH'][3]['score'])


# In[67]:


print(summaries['ETH'][3])
print(scores['ETH'][3]['score'])
print(scores['ETH'][3]['label'])


# # 6. Exporting results to CSV

# In[68]:


summaries


# In[69]:


scores


# In[70]:


cleaned_urls


# In[71]:


range(len(summaries['ETH']))


# In[72]:


summaries['ETH'][3]


# In[73]:


def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]
            output.append(output_this)
    return output


# In[74]:


final_output = create_output_array(summaries, scores, cleaned_urls)
final_output


# In[75]:


# Insert a header - column names to the csv file
final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL'])


# In[76]:


final_output


# In[77]:


import csv
with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)


# In[ ]:




