Python文本数据分析：新闻分类任务

【Packages】
jieba
pandas
wordcloud
matplotlib
sklearn

【概念】
IDF：Inverse Document Frequency 逆文档频率
TF-IDF=词频(TF) X 逆文档频率(IDF)
词频(TF) = 某个词在文章中的出现次数/该文出现次数最多的词的出现次数
逆文档频率(IDF) = log(语料库的文档总数/(包含该词的文档数+1))

【步骤】
1、去停用词
2、TF-IDF关键词提取
3、LDA建模
4、贝叶斯算法


