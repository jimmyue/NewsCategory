#!/usr/bin/python3
# -*- coding:utf-8 -*-
'''
Created on 2021年2月9日
@author: jimmy
'''
import jieba
import jieba.analyse
import pandas as pd 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB #贝叶斯模型

def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append({'word':word, 'count':1})
        contents_clean.append(line_clean)
    return contents_clean,all_words

#读取数据
df_news=pd.read_table('./data/data.txt',names=['category','theme','url','content'],encoding='utf-8')
df_news=df_news.dropna()
#转为List
content=df_news.content.values.tolist()
#jieba分词
contents=[]
for line in content:
	current_segment=jieba.lcut(line)
	if len(current_segment)>1 and current_segment!='\r\n':
		contents.append(current_segment)
#停用词
stopword=pd.read_table('./data/stopwords.txt',index_col=False,sep='\t',quoting=3,names=['stopwords'],encoding='utf-8')
stopwords = stopword.stopwords.values.tolist()
#去停用词
contents_clean,all_words = drop_stopwords(contents,stopwords)

# #计算词频
# df_all_words=pd.DataFrame(all_words)
# words_count=df_all_words.groupby('word')['count'].sum()
# words_count=words_count.reset_index().sort_values(by=["count"],ascending=False)
# #词云展示
# matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
# wordcloud=WordCloud(font_path="./data/simhei.ttf",background_color="white",max_font_size=80)
# word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
# wordcloud=wordcloud.fit_words(word_frequence)
# plt.imshow(wordcloud)
# plt.show()

df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
label=df_train.label.unique().tolist()
label_mapping={}
for i in range(len(label)):
	label_mapping[label[i]]=i
df_train['label'] = df_train['label'].map(label_mapping) #构建一个映射方法
#将原始数据按照比例分割为“测试集”和“训练集”
x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)
words = []
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        print (line_index,word_index)

#1.词袋模型的特征来建模
#制作词袋模型特征
vec = CountVectorizer(analyzer='word', max_features=4000,  lowercase = False)
feature = vec.fit_transform(words)
#使用词袋模型的特征来建模，观察结果
classifier = MultinomialNB() 
classifier.fit(feature, y_train)
test_words = []
for line_index in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index,word_index)
print(classifier.score(vec.transform(test_words), y_test))
#2.TF-IDF特征建模
#制作词袋模型特征
vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vectorizer.fit(words)
#使用TF-IDF特征建模，观察结果
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)
print(classifier.score(vectorizer.transform(test_words), y_test))