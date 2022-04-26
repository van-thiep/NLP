# Overview
This part explains all modern word embedding algorithms

Each part trys to answer the following questions:
- How it works? 
- What is pros and cons/ applications?
- Best practice?
# TF-IDF 
## Algos
**Tf-idf** is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. The vectorization process is similar to **Bag-of-word** technique.Alternatively, the value corresponding to the word is assigned a **tf-idf** value instead of the number of the occurrence of word. The tf-idf value is obtained by multiplying the TF and IDF values.

**TF(Term frequency)** measures how often a word occurs in a document. Since different documents in the corpus may be of different length, a term may occur more often in a longer document as compared to a shorter ducument. To normalize these counts, we divide the number of occurrences by the length of document. TF of a term t in a document d is defined as: 

<img src="https://render.githubusercontent.com/render/math?math=TF(t,d) = \frac{Number of occurrences of term t in document d}{Total number of terms in document d}">

**IDF (inverse document frequency)** measures the importance of a term across a corpus. In computing TF, all term are given equal importance (weightage). It's a well-known fact that stop words like is,am, a, an ,the,... are not important, even though they occur frequently. To account for such cases, IDF weighs down the terms that are very common across a corpus and weigh up rare terms. IDF of a term is calculated as following:

<img src="https://render.githubusercontent.com/render/math?math=TF(t,d) = log_e \frac{Total number of document in corpus}{Number of document with term t in them}">

## Pros and cons 
Disadvantage 
- It still relies on lexical analysis and does not take into account things such as the co-occurrence of terms, semantics, the context associated with terms, and the position of a term in a document. It is dependent on the vocabulary size, like *CountVectorizer* , and will get really slow with large vocabulary sizes.
- The feature vectors are sparse and high-dimensional representations --> make them computationally inefficient.
- They cannot handle OOV words.
## References
- https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- https://towardsdatascience.com/word-embedding-techniques-word2vec-and-tf-idf-explained-c5d02e34d08
# Word2vec 
# Glove 
# Fasttext 
# Transfomer