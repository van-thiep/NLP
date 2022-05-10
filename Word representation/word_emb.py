
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
import string
import spacy 
import en_core_web_sm
from gensim.models import Word2Vec, keyedvectors
from gensim.test.utils import common_texts
import pickle

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
        """
        Take documents as inputs and return list of tokens.
        """
        stemmer = SnowballStemmer('english')
        tokens = nltk.word_tokenize(sentence)
        filtered_tokens = [t for t in tokens if t not in string.punctuation]
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems

class word2vec:
    def __init__(self) -> None:
        pass

    def pretrained_spaCy(self, sent:str):
        """
        Using spaCy's pretrained model
        """
        # Load pretrained model 
        nlp = en_core_web_sm.load()
        doc = nlp(sent)
        print(doc.vector) 

    def pretrained_gensim(self, pretrained_path, word:str):
        # load pretrained model 
        w2v_model = keyedvectors.load_word2vec_format(pretrained_path,binary=True)
        print(len(w2v_model.vocab)) # Number of words in vocabulary 
        print(w2v_model.most_similar[word])
        # Return the embedding of word
        return w2v_model[word]
    
    def train(self, w2v_model_path):
        # Defind the model by selecting the parameters. 
        our_model = Word2Vec(common_texts, size=10, 
                             window=5, min_count=1, workers=4)
        # Train model 
        our_model.train(common_texts, total_examples=len(common_texts),
                         epochs=200)
        # Save model
        pickle.dump(our_model, open(w2v_model_path,'wb'))
        # Inspect the model by looking the most similar words for a test word
        print(our_model.wv.most_similar('computer',topn=5))
        # Let see 10-dimension embedding of computer
        print(our_model['computer'])
        return our_model