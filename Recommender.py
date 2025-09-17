# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_split
import requests, io, zipfile

# ------------------------------------------------------------
# 1. LOAD ONE DATA SOURCE  (MovieLens 100K) ------------------------------
# ------------------------------------------------------------
url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
z = zipfile.ZipFile(io.BytesIO(requests.get(url).content))

ratings = pd.read_csv(z.open('ml-100k/u.data'),
                      sep='\t', header=None,
                      names=['user','item','rating','timestamp'])
movies = pd.read_csv(z.open('ml-100k/u.item'),
                     sep='|', header=None, encoding='latin-1',
                     names=['item','title']+[f'x{i}' for i in range(8)])
movies['genres'] = movies.iloc[:, 3:23].apply(lambda x: ' '.join(x.index[x==1]), axis=1)
# merge
df = ratings.merge(movies[['item','title','genres']], on='item')

# ------------------------------------------------------------
# 2. CONTENT-BASED  (TF-IDF on genres) ----------------------------------
# ------------------------------------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_idx = pd.Series(movies.index, index=movies['item']).drop_duplicates()

def recommend_content(user_id, N=5):
    # pick user's top-rated movies
    user_movies = df[df.user==user_id].sort_values('rating', ascending=False).head(10)
    idx = user_movies['item'].map(movie_idx)
    scores = cosine_sim[idx].mean(axis=0)
    top_idx = scores.argsort()[::-1]
    # remove already seen
    seen = set(user_movies['item'])
    recs = movies.iloc[top_idx][~movies['item'].isin(seen)].head(N)
    return recs[['item','title']]

# ------------------------------------------------------------
# 3. USER-BASED COLLABORATIVE (k-NN on users) ----------------------------
# ------------------------------------------------------------
pivot_u = df.pivot(index='user', columns='item', values='rating').fillna(0)
knn_u = NearestNeighbors(metric='cosine', n_neighbors=20)
knn_u.fit(pivot_u)

def recommend_user(user_id, N=5):
    # find similar users
    dist, neigh = knn_u.kneighbors(pivot_u.loc[user_id].values.reshape(1,-1))
    similar_users = pivot_u.index[neigh[0][1:]]  # drop self
    # aggregate their ratings
    candidate = pivot_u.loc[similar_users].mean(axis=0).sort_values(ascending=False)
    seen = pivot_u.loc[user_id].nonzero()[0]
    candidate.iloc[seen] = np.nan
    top_items = candidate.dropna().head(N).index
    return movies[movies['item'].isin(top_items)][['item','title']]

# ------------------------------------------------------------
# 4. ITEM-BASED COLLABORATIVE (k-NN on items) -----------------------------
# ------------------------------------------------------------
pivot_i = pivot_u.T
knn_i = NearestNeighbors(metric='cosine', n_neighbors=20)
knn_i.fit(pivot_i)

def recommend_item(user_id, N=5):
    user_ratings = pivot_u.loc[user_id]
    seen = user_ratings.nonzero()[0]
    # for each seen item find neighbours and score by rating
    scores = np.zeros(pivot_i.shape[0])
    for item_idx in seen:
        rating = user_ratings.iloc[item_idx]
        if rating==0: continue
        dist, neigh = knn_i.kneighbors(pivot_i.iloc[item_idx].values.reshape(1,-1))
        for n_idx in neigh[0][1:]:
            scores[n_idx] += rating * (1/(1+dist[0][1:]))[np.where(neigh[0]==n_idx)][0]
    top_items = pd.Series(scores, index=pivot_i.index).sort_values(ascending=False).head(N).index
    return movies[movies['item'].isin(top_items)][['item','title']]

# ------------------------------------------------------------
# 5. HYBRID  (50 % content score + 50 % item-based score) ----------------
# ------------------------------------------------------------
def recommend_hybrid(user_id, N=5):
    cb  = recommend_content(user_id, N=20)
    itb = recommend_item(user_id, N=20)
    # simple weighted rank fusion
    cb['score']  = 0.5 * (len(cb) - cb.index)
    itb['score'] = 0.5 * (len(itb) - itb.index)
    combined = pd.concat([cb, itb]).groupby(['item','title'], as_index=False)['score'].sum()
    return combined.sort_values('score', ascending=False).head(N)

# ------------------------------------------------------------
# 6. MATRIX-FACTORISATION benchmark (Surprise SVD) ------------------------
# ------------------------------------------------------------
reader = Reader(rating_scale=(1,5))
surprise_data = Dataset.load_from_df(df[['user','item','rating']], reader)
train, test = surprise_split(surprise_data, test_size=.25, random_state=42)
svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
svd.fit(train)
print("SVD test RMSE:", accuracy.rmse(svd.test(test)))

def recommend_svd(user_id, N=5):
    # all unseen items
    all_items = movies['item'].unique()
    seen = set(df[df.user==user_id]['item'])
    preds = [svd.predict(user_id, iid) for iid in all_items if iid not in seen]
    top = sorted(preds, key=lambda x: x.est, reverse=True)[:N]
    return pd.DataFrame([(p.iid, movies[movies.item==p.iid].title.iloc[0]) for p in top],
                        columns=['item','title'])

# ------------------------------------------------------------
# 7. QUICK DEMO  (user #196) ---------------------------------------------
# ------------------------------------------------------------
user = 196
print("Content-Based :")
print(recommend_content(user))
print("\nUser-Based CF :")
print(recommend_user(user))
print("\nItem-Based CF :")
print(recommend_item(user))
print("\nHybrid        :")
print(recommend_hybrid(user))
print("\nSVD (MF)      :")
print(recommend_svd(user))
