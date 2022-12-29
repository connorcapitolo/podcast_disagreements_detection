import numpy as np
import matplotlib.pyplot as plt
from common.transcript_utils import get_tokens, get_avg_word_vector, get_cosine_similarities, process_transcript  

class SBERTVectorizer:
    def __init__(self, model, truncate = False, normalize = False):
        self.model = model
        self.truncate = truncate
        self.normalize = normalize
    
    def vectorize(self, segment_text_list):
        """
        Given a list of text segments, return:
            segment_vectors: a np array of np arrays (each entry is the embedding for each segment)
            filter_mask: None
        """
        # by default, the sentence transformer truncates to the first 128 word pieces
        if self.truncate:
            return self.model.encode(segment_text_list, 
                                     normalize_embeddings = self.normalize), np.ones(len(segment_text_list))
        else: 
            batch_size = 128; embedding_size = 384
            segment_vectors = []

            # Handle sequences of longer than 128 words by weighted avg of 128-length embeddings
            for segment in segment_text_list:
                batch_vectors = []; weights = []
                for i in range(0, len(segment), batch_size):
                    batch = segment[i:i+batch_size]
                    weights.append(len(batch))
                    batch_vectors.append(self.model.encode(batch, normalize_embeddings = self.normalize))
                segment_vectors.append(np.average(np.vstack(batch_vectors), 
                                                  axis = 0,
                                                  weights = weights))
            return np.array(segment_vectors), np.ones(len(segment_text_list))

class WordEmbeddingVectorizer:
    def __init__(self, embedding_dict, embedding_dim, lemmatize, remove_stopwords, n_token_filter):
        self.embedding_dict = embedding_dict
        self.embedding_dim = embedding_dim
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.n_token_filter = n_token_filter
    
    def vectorize(self, segment_text_list):
        """
        Given a list of text segments, return:
            segment_vectors: a np array of np arrays (each entry is the averaged embedding for each segment)
            filter_mask: binary np array with 0s where there are < than n_token_filter tokens (for a given segment)
        """
        segment_vectors = []; filter_mask = []
        for segment in segment_text_list:
            segment_tokens = get_tokens(segment, self.lemmatize, self.remove_stopwords)
            segment_vector, n_tokens = get_avg_word_vector(segment_tokens, self.embedding_dict, self.embedding_dim)
            filter_mask.append(n_tokens >= self.n_token_filter)
            segment_vectors.append(segment_vector)
        return np.array(segment_vectors), np.array(filter_mask)

class Transcript:
    def __init__(self, json_filepath, segment_vectorizer = None, segment_definition = 'default'):
        self.json_filepath = json_filepath
        self.data = process_transcript(json_filepath, segment_definition = segment_definition)
        self.segment_vectorizer = segment_vectorizer
        self.segment_vectors = None
        self.filter_mask = None
        self.segment_sentiments = None
        self.segment_sentiments_scores = None
        
    def __repr__(self):
        return f"{self.__class__.__name__} with variables: {list(self.__dict__.keys())}"
                 
    def set_segment_vectors(self, segment_vectorizer):
        self.segment_vectors, self.filter_mask = segment_vectorizer.vectorize(self.data['text_chunks'])
        
    def set_sentiments(self, classifier):
        """
        Given a classifier, initialize:
            sentiments: list of sentiments (1 = positive) corresponding to each audio segment
            sentiments_scores: scores returned from the classifier for each audio segment
        """
        sentiments = []; scores = []
        for t in self.data['text_chunks']:
            classification = classifier(t)
            assert(len(classification)==1) # validate that classifier returns a single classification
            sentiments.append(1*(classification[0]['label'] == "POSITIVE"))
            scores.append(classification[0]['score'])
        self.segment_sentiments = np.array(sentiments)
        self.segment_sentiments_scores = np.array(scores)
        
    def plot_pca(self, remove_invalid = True, title = "PCA", n_chars = 15):
        """
        Plot segment vectors via PCA, using sentiments as colors and labeling with the text of the segment
        remove_invalid: 
            if set to True, filters out segment vectors using self.filter_mask
            True by default because the vectors may be bad if number of tokens in a segment is low or zero
        title: the title of the plot
        n_chars: the number of characters of each segment text to display in the plot
        """ 
        assert(len(self.segment_vectors)==len(self.segment_sentiments))
        assert(len(self.segment_sentiments)==len(self.data['text_chunks']))
        filter_mask = self.filter_mask if remove_invalid else np.array([True]*len(self.filter_mask))
            
        segment_vectors = self.segment_vectors[self.filter_mask]
        segment_sentiments = self.segment_sentiments[self.filter_mask]
        segment_text = np.array(self.data['text_chunks'])[self.filter_mask]
            
        pca = PCA(n_components=2)
        X = pca.fit_transform(segment_vectors)
        plt.figure(figsize = (20,10))
        plt.scatter(X[:,0], X[:,1], c = np.array(['red', 'blue'])[segment_sentiments])
        for i, txt in enumerate(segment_text):
            plt.annotate(txt[:n_chars], (X[:,0][i], X[:,1][i]))

        # https://stackoverflow.com/questions/39500265/manually-add-legend-items-python-matplotlib
        red_patch = mpatches.Patch(color='red', label='negative sentiment')
        blue_patch = mpatches.Patch(color='blue', label='positive sentiment')
        plt.legend(handles=[red_patch, blue_patch])
        plt.title(title)
        
        
    def get_disagreement_cosine(self, query_text, cos_similarity_threshold, remove_invalid = True):
        """
        Return text, times, cos similarity of possible disagreement segments
        remove_invalid: 
            if set to True, filters out segment vectors using self.filter_mask
            True by default because the vectors may be bad if number of tokens in a segment is low or zero
        """
        # convert query to tokens
        q, _ = self.segment_vectorizer.vectorize([query_text])
        q = q[0]

        cosine_similarities = get_cosine_similarities(q, self.segment_vectors)
        output_mask = (cosine_similarities > cos_similarity_threshold)
        if remove_invalid: output_mask = (output_mask)&(self.filter_mask)

        disagreement_text = np.array(self.data['text_chunks'])[output_mask]
        disagreement_times = np.array(self.data['text_chunks_times'])[output_mask]
        cosine_similarities = cosine_similarities[output_mask]
        return disagreement_text, disagreement_times, cosine_similarities
