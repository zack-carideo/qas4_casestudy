## BASIC PROFILES
#pre_filt_df = pd.read_csv(data_path.replace('.zip',''))
#pre_filt_df = pre_filt_df[~pre_filt_df[text_col].isna()].copy()
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


def plot_target_dist(df: pd.DataFrame, target_col:str = 'product_group'):

    p_tot = df[target_col].value_counts()/len(df)
    labels = p_tot.index
    fig, ax = plt.subplots()
    ax.barh([idx for idx,lab in enumerate(labels)],p_tot, align='center')
    ax.set_yticks([idx for idx,lab in enumerate(labels)])
    ax.set_yticklabels( labels)
    ax.invert_yaxis()

    plt.show()


def get_tfidf_vectorizer(ngram_range=(1,3),min_df = 2, max_df = .2 , max_features = None, sublinear_tf = True, binary=False, smooth_idf = True):
    return TfidfVectorizer(ngram_range = ngram_range
    , tokenizer = None
    , min_df = min_df
    , max_df = max_df 
    , lowercase = True
    , analyzer = 'word'
    , max_features = max_features
    , binary = binary 
    , use_idf = True
    , smooth_idf = smooth_idf 
    , sublinear_tf = sublinear_tf
    , stop_words = 'english'
    )


def topn_tfidf_freq(df:pd.DataFrame, text_col: str, n=20):
    """return topn tfidf weighted terms for a series(use this stratifed by target level) """
    vec = get_tfidf_vectorizer()
    ft_vec = vec.fit_transform(df[text_col].dropna())
    occ=np.asarray(ft_vec.mean(axis=0)).ravel().tolist()
    return pd.DataFrame({'term':vec.get_feature_names(), 'weight':occ}).sort_values(by='weight', ascending=False)


def run_eda(df, text_colname, target):
    # view distribution of target
    plot_target_dist(df)

    # view distribution of text by product group
    df[f'{text_colname}_len'] = df[text_colname].apply(lambda x: len(x))

    # describe it all
    df.groupby(target)[f'{text_colname}_len'].describe()

    # view length of text by product group
    df[[target, f'{text_colname}_len']].groupby(target).plot(kind='kde', ax=plt.gca())


def tfidf_visuals(tf_idf_matrix
                  , num_clusters=7
                  , num_seeds=10
                  , max_iterations=100
                  , pca_num_components = 2
                  , tsne_num_components=2
                  , labels_color_map = {0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'}                                       
                  , tsne = False
                 ):
    """
    tf_idf_matrix: a fit sklearn.tfidf_vectorizer
    """
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    

    # create k-means model with custom config
    clustering_model = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations,
    )
    
    labels = clustering_model.fit_predict(tf_idf_matrix)
    print('cluster done')
    X = tf_idf_matrix.todense()
    
    # ----------------------------------------------------------------------------------------------------------------------
    
    reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
    # print reduced_data
    print('pca done')

    fig, ax = plt.subplots()
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
        print(labels[index])
    plt.show()
    
    if tsne ==True:
        # t-SNE plot
        embeddings = TSNE(n_components=tsne_num_components)
        Y = embeddings.fit_transform(X)
        plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
        plt.show()

    return 'plot complete'