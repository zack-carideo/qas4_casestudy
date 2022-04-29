import pandas as pd
import gensim, re
from gensim.utils import simple_preprocess, lemmatize
from os import path

def generate_StopWords_Phrases(stop_list):
    """

    @param stop_list: list of strings
    @return:
    """
    stop_terms = []
    stop_phrases = []
    for item in list(stop_list):
        if len(item.split())==1:
            stop_terms.append(item.lower())
        if len(item.split())>1:
            stop_phrases.append(' '.join([word.lower() for word in item.split()]))
    return stop_terms, stop_phrases

def encode_target_alt(data:pd.DataFrame , target='LegalAction'):
    "factorize target and return mapping of index to actual value"
    factor = pd.factorize(data[target])
    transvals = factor[0]
    definitions = factor[1]

    data['label_id'] = transvals
    data = data.drop(columns=target)
    return data, definitions



def remove_stopwords_string(text,stop_words =None,min_len=2):
    if stop_words is None:
        stop_words = []
    return ' '.join([word for word in simple_preprocess(str(text)) if word not in stop_words and len(word)>min_len])

def remove_nonchars(text):
    return ' '.join([re.sub('[^A-Za-z|^\$|^\.]+', ' ', word) for word in text.split(' ') if (word.isalnum() and len(word)>2)])

def replace_stopphrases(doc,phrases):
    for item in phrases:
        if item in doc.lower():

            doc = doc.lower().replace(item,' ')
    return doc


def incorp_phrases(texts, stop_words=None):
    if stop_words is None:
        stop_words = []
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc), deacc=True) if word not in stop_words] for doc in texts]

    """generate bi and tri grams"""
    bigram = gensim.models.Phrases(texts, min_count=10, threshold=10)
    trigram = gensim.models.Phrases(bigram[texts], threshold=30)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    return texts

def text_clean(data,textcol='text',stop_phrases=None,stop_list=None):
    """

    @param Data: pd.dataframe
    @param textcol: name of column with string of txt
    @param stop_phrases: list of stop phrases to remove
    @param stop_list: list of stop words + stop_phrases
    @return: pd.series
    """
    #remove consequivtly captialized letters and make them cap case and remove stop phrases (must remove stop phrases before stop words)
    #then: remove numbers -> remove stopwords & phrases-> remove excluded punct
    data[textcol] = data[textcol].apply(lambda x:  ' '.join([word.title()  if (word.isupper() and word.lower() not in stop_list) else
        word for word in remove_stopwords_string(
                remove_nonchars(
                        replace_stopphrases(
                                str(x),stop_phrases
                                ))).split()
        ]))

    return data[textcol]


def lemawork():
    try:
        lemmatize('disputed inaccurate information requested look')
    except:
        pass

def lemma_wordtok_text(text_str,allowed_postags=['NN','VB','JJ','RB'],min_length=3):
    """

    @param word_toks:
    @param allowed_postags: by default NN , ADJ , JJ, VB, RB tags returned
    @param min_length:
    @return:
    """
    try:
        lemawork()
    except:
        lemmatize('disputed inaccurate_information requested look')

    text = [word.decode('utf-8').split('/')[0] for word in lemmatize(text_str
                                                                     #, allowed_tags=re.compile(allowed_postags)
                                                                     ,min_length=min_length)
            ]
    return text


# get dictionary of sequences and occurrences of words for each string
# input is a column containing word tokenized string (ex. ['I','am','happy'])
def get_doc2bow(Data, text_col, filter_extrms=True, low_doc_lim=10, upper_doc_perc=.4, maxterms=None):
    """
    Data : Dataframe
    text_col: str (ex. 'final_text_col') , referencing a column containing observations of word tokenized strings (first order list)
    filter_extrms: boolean (if True, each string will be filtered to only include the values captured within the filtering critera)
    low_doc_lim: int (only used if filter_extrms ==True, indicates the lowest number of documents a term must be found in to consider in output)
    upper_doc_perc: float (only used if filter_extrms==True, indicates the max percentage of documents containing a word, if p(word)>upper_doc_perc then the term is filtered out of output)
    """
    input_text = Data[text_col]

    # create dictonary mapping of each word in text
    id2word = gensim.corpora.Dictionary(input_text)

    # filter common and very rare words, and limit total words to use in final output list of strings (word tokenized list of words)
    if filter_extrms == True:
        id2word.filter_extremes(no_below=low_doc_lim, no_above=upper_doc_perc, keep_n=maxterms)

        # remove gaps in id sequence after words taht were removed
        id2word.compactify()

    # for each document create a dictonary reporting how many times each word appears in string
    bow_corpus = [id2word.doc2bow(doc, allow_update=False) for doc in input_text]

    return id2word, bow_corpus


def preprocess_data(df
                              ,text_colname
                              , stop_phrases
                              , stop_list
                              , keras_seq_flag = False
                              , project_folder =   None
                              ):

    assert project_folder is not None, 'please provide location to save model output and summary metrics'

    #remove censored words if NOT using a sequence based model input
    if keras_seq_flag is not True:
        df[text_colname] = df[text_colname].apply(
            lambda x: ' '.join([word for word in str(x).split() if word.count('X') < 3]))


    #clean text
    df[text_colname]= text_clean(df,textcol=text_colname,stop_phrases=stop_phrases, stop_list = stop_list)
    df.dropna(subset=[text_colname],inplace=True)

    #incorporate common phrases into text
    df[text_colname] = incorp_phrases(df[text_colname], stop_words=stop_list)
    df.dropna(subset=[text_colname],inplace=True)

    #subset dataframe to only include rows with at least 10 words
    mask = (df[text_colname].str.len()>20)
    df = df.loc[mask]

    #lemmatize text (instead of stemming to retain meaning for when using embeddings, and for human readabilty)
    #only consider NN to try and reduce training time  with minimal information loss by reducing noise
    try:
        df[text_colname]  = df[text_colname].apply(lambda x: lemma_wordtok_text(' '.join([word for word in x]),min_length=1))
    except:
        df[text_colname]  = df[text_colname].apply(lambda x: lemma_wordtok_text(' '.join([word for word in x]),min_length=1))

    df.dropna(subset=[text_colname],inplace=True)

    # create doc2bow and id2word mapping and filter extreme values
    # NOTE: We could do this with tfidf vectorizor, but doing it like this preserves the input string)
    id2word,bow_corpus = get_doc2bow(df
                                     , text_colname
                                     , filter_extrms=True
                                     , low_doc_lim =10
                                     , upper_doc_perc=.4
                                     , maxterms = None)

    #if filtering is applied subset original texts to only include the words present after filtering
    #EVEN IF USING EMBEDDINGS, FILTERING HERE TO TRY TO REDUCE RESOURCES REQUIRED TO TRAIN MODEL
    vocab= list(set([item for sublist in [[id2word[key[0]] for key in bow_corpus[i]] for i in range(0,len(df))] for item in sublist]))
    texts = [[word for word in doc if word in vocab]for doc in df[text_colname]]
    df['final_complaint_text']=texts

    #tfidf requies inputs to be string not word tokens, so we join the tokens into a string
    df['final_complaint_string']= df['final_complaint_text'].apply(lambda x: ' '.join([term for term in x]))
    #df.drop(columns = ['final_complaint_text'])

    #must rename columns to remove _ for sklearn pipelines
    df = df.rename(columns={col:col.replace('_','') for col in list(df)})

    #save file after all preprocessing
    if path.exists(project_folder+'cfpb_postbowfilter.pickle')==False:
        pd.to_pickle(df,project_folder+'cfpb_postbowfilter.pickle')

    return df
