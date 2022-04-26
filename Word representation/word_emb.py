
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
import string

# TFIDF 
class tfIdf:
    def __init__(self) -> None:
        pass 
    def train(self, train_data:list, stopword_list:list):
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        # https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
        self.tfidf_vectorizer = TfidfVectorizer(
                            analyzer='word', #'char_wb'
                            max_df=0.90, 
                            max_features=200000,
                            min_df=0.05, 
                            stop_words=stopword_list,
                            use_idf=True,
                            tokenizer=self.tokenize_and_stem,
                            ngram_range=(1,3)
                            )
        self.tfidf_vectorizer = self.tfidf_vectorizer.fit(train_data)
    
    def infer(self,sent):
        tfidf_matrix = self.tfidf_vectorizer.transform(sent)
        return tfidf_matrix
    
    def interpretation(self):
        print(self.tfidf_vectorizer.get_feature_names())
    
    def tokenize_and_stem(self,sentence):
        stemmer = SnowballStemmer('english')
        tokens = nltk.word_tokenize(sentence)
        filtered_tokens = [t for t in tokens if t not in string.punctuation]
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems