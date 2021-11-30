from django.shortcuts import render
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt ; plt.rcdefaults() 
import pandas as pd
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
import warnings
import numpy as np
import io
import os

warnings.filterwarnings("ignore")


# Create your views here.

def readcsv():
    x = pd.read_csv('/home/kali/dataset/movie_dataset.csv')
    return x

def home(request):
    return render(request,'app1/home.html')

def about(request):
    return render(request, 'app1/about us.html')    

def demo(request):
    
    df = readcsv()

    C= df['vote_average'].mean() #mean vote

    m = df['vote_count'].quantile(0.9) 

    q_movies = df.loc[df['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)

    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

    q_movies = q_movies.sort_values('score', ascending=False)

    movies = q_movies[['title', 'director', 'genres', 'score']].head(10)

    set1=list(movies['title'])
    set2=list(movies['director'])
    set3=list(movies['genres'])
    set4=list(movies['score'])

    para4=zip(set1,set2,set3,set4)


    context={
        'para1' : para4,
    }
    
    return render(request,'app1/demo.html', context) 

def visual(request):
    df = readcsv()

    C= df['vote_average'].mean()

    m = df['vote_count'].quantile(0.9)

    q_movies = df.loc[df['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        
        return (v/(v+m) * R) + (m/(m+v) * C)

    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

    q_movies = q_movies.sort_values('score', ascending=False)

    fig =Figure()
    canvas= FigureCanvas(fig)
    pop = q_movies.sort_values('score', ascending=False)
    plt.figure(figsize=(14,6))
    plt.barh(pop['title'].head(6),pop['score'].head(6), align='center',
            color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Ratings")
    plt.title("Most Rated Movies")
    buf = io.BytesIO()
    plt.savefig(buf,format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response

def visual2(request):
    df = readcsv()
    C= df['vote_average'].mean()

    m = df['vote_count'].quantile(0.9)

    q_movies = df.loc[df['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        
        return (v/(v+m) * R) + (m/(m+v) * C)

    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

    q_movies = q_movies.sort_values('score', ascending=False)
    
    fig =Figure()
    canvas= FigureCanvas(fig)
    pop = q_movies.sort_values('popularity', ascending=False)
    plt.figure(figsize=(14,6))
    plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
            color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Popularity")
    plt.title("Most Popular Movies")
    buf = io.BytesIO()
    plt.savefig(buf,format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response

def visual3(request):
    df = readcsv()
    fig=Figure()
    canvas= FigureCanvas(fig)
    plt.figure(figsize=(14,6))
    most_movies = df.groupby('director')['title'].count().reset_index().sort_values('title',ascending=False).head(10)
    dic_name = most_movies['director']
    index = np.arange(len(dic_name))
    plt.bar(index, most_movies['title'], align='center', color='skyblue')
    plt.xlabel('Directors', fontsize=20)
    plt.ylabel('No of Movies', fontsize=20)
    plt.xticks(index, most_movies['director'], fontsize=8, rotation=30)
    plt.title('Top 10 directors with most movies',fontsize=20)
    buf = io.BytesIO()
    plt.savefig(buf,format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response

def content(request):
    return render(request,'app1/content based.html')

def content_1(request):
    return render(request,'app1/content_1.html')

def collaboritive(request):
    return render(request, 'app1/collaboritive filtering.html')


def showcontent_1(request):
    u_movies = request.GET.get('mname','default')

    if u_movies != "":

        df=pd.read_csv('/home/kali/dataset/movie_dataset.csv', error_bad_lines=False)
        print('Size of data frame is :',df.shape)
        df.index=df['index']
       
        trial = df[['vote_average', 'vote_count']]
        data = np.asarray([np.asarray(trial['vote_average']), np.asarray(trial['vote_count'])]).T

        #we'll use the Elbow Curve method for the best way of finding the number of clusters for the data
        #From the elbow plot, we get that the elbow lies around the value K=5, so that's what we will attempt it with
        #Computing K means with K = 5, thus, taking it as 5 clusters
        centroids, _ = kmeans(data, 5)

        #assigning each sample to a cluster
        #Vector Quantisation:

        #idx, _ = vq(data, centroids)


        def segregation(data):
            values = []
            for val in data.vote_average:
                if val>=0 and val<=1:
                    values.append("Between 0 and 1")
                elif val>1 and val<=2:
                    values.append("Between 1 and 2")
                elif val>2 and val<=3:
                    values.append("Between 2 and 3")
                elif val>3 and val<=4:
                    values.append("Between 3 and 4")
                elif val>4 and val<=5:
                    values.append("Between 4 and 5")  
                else:
                    values.append("NaN")
            return values
        df['Ratings_Dist'] = segregation(df)
        books_features = pd.concat([df['Ratings_Dist'].str.get_dummies(sep=","), df['vote_average'], df['vote_count']], axis=1)
        '''
        The min-max scaler is used to reduce the bias which would have been present due to some books having a 
        massive amount of features, yet the rest having less. 
        Min-Max scaler would find the median for them all and equalize it.
        '''
        min_max_scaler = MinMaxScaler()
        books_features = min_max_scaler.fit_transform(books_features)                                    
        model = neighbors.NearestNeighbors(n_neighbors=11, algorithm='ball_tree',metric='euclidean')

        model.fit(books_features)
        distance, indices = model.kneighbors(books_features)
        '''
        Creating specific functions to help in finding the book names:
        Get index from Title
        Get ID from partial name (Because not everyone can remember all the names)
        Print the similar books from the feature dataset. (This uses the Indices metric from the nearest neighbors to pick the books.)
        '''

        similarBooks=list()
        partialNameId=list()

        def get_index_from_name(name):
            return df[df["title"]==name].index.tolist()[0]

        all_books_names = list(df.title.values)

        def get_id_from_partial_name(partial):
            partialNameId=[]
            for name in all_books_names:
                if partial in name:
                        partialNameId.append(name+str(all_books_names.index(name)))
            return partialNameId

        def print_similar_books(query=None,id=None):
            similarBooks=[]
            if id:
                for id in indices[id][1:]:
                    similarBooks.append(df.iloc[id]["title"])
                return similarBooks
            if query:
                found_id = get_index_from_name(query)
                for id in indices[found_id][1:]:
                    similarBooks.append(df.iloc[id]["title"])
                return similarBooks

        lol_2 = print_similar_books(u_movies)

        context2 ={
            'pa2':lol_2
        }

        return render(request, "app1/content2res.html", context2)

    else:

        return HttpResponse("PLEASE TYPE A MOVIE NAME")
    



            
def showcontent(request):
    user_liked_movie = request.GET.get('mname', 'default')

    if user_liked_movie != "":

        def content_based(movie_user_likes):
            
            
            def get_title_from_index(index):
                return df[df.index == index]["title"].values[0]

            def get_index_from_title(title):
                return df[df.title == title]["index"].values[0]
            
            df = readcsv()


            features = ['keywords','cast','genres','director']

            for feature in features:
                df[feature] = df[feature].fillna('')
            

            def combine_features(row):
                return row['keywords'] +" "+row['cast'] +" "+ row['genres'] +" "+ row['director']


            df["combined_features"] = df.apply(combine_features,axis=1)

            cv = CountVectorizer()

            count_matrix = cv.fit_transform(df["combined_features"])

            cosine_sim = cosine_similarity(count_matrix)

            movie_index = get_index_from_title(movie_user_likes)

            similar_movies = list(enumerate(cosine_sim[movie_index])) 

            sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

            name = [] 

            i=0
            for movie in sorted_similar_movies:
                name.append(get_title_from_index(movie[0]))
                i=i+1
                if i>10:
                    break
            
            return name    
            

        lol = content_based(user_liked_movie)

        context={
            'pa1' : lol,
        }
        return render(request, "app1/contentres.html", context)

    else:

        return HttpResponse("PLEASE TYPE A MOVIE NAME")

def showcollaborative(request):

    my_favorite = request.GET.get('mname', 'default')
    if my_favorite != "":

        data_path = '/home/kali/dataset/'
        movies_filename = 'movies.csv'
        ratings_filename = 'ratings.csv'

        df_movies = pd.read_csv(os.path.join(data_path, movies_filename),usecols=['movieId', 'title'],dtype={'movieId': 'int32', 'title': 'str'})

        df_ratings = pd.read_csv(os.path.join(data_path, ratings_filename),usecols=['userId', 'movieId', 'rating'],dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        df_ratings=df_ratings[:2000000]

        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

        num_users = len(df_ratings.userId.unique())
        num_items = len(df_ratings.movieId.unique())
        print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))

        df_ratings_cnt_tmp = pd.DataFrame(df_ratings.groupby('rating').size(), columns=['count'])

        total_cnt = num_users * num_items
        rating_zero_cnt = total_cnt - df_ratings.shape[0]

        df_ratings_cnt = df_ratings_cnt_tmp.append(pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),verify_integrity=True).sort_index()

        df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])

        df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])

        #now we need to take only movies that have been rated atleast 50 times to get some idea of the reactions of users towards it

        popularity_thres = 50
        popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
        df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
            
        df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])

        ratings_thres = 50
        active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
        df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]

        #pivot and create movie-user matrix
        movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        #map movie titles to images
        movie_to_idx = {
            movie: i for i, movie in 
            enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
        }
        # transform matrix to scipy sparse matrix
        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

        # define model
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

        def fuzzy_matching(mapper, fav_movie, verbose=True):
            """
            return the closest match via fuzzy ratio. 
        
            Parameters
            ----------    
            mapper: dict, map movie title name to index of the movie in data
            fav_movie: str, name of user input movie
        
            verbose: bool, print log if True
            Return
            ------
            index of the closest match
            """
            match_tuple = []
            # get match
            for title, idx in mapper.items():
                ratio = fuzz.ratio(title.lower(), fav_movie.lower())
                if ratio >= 60:
                    match_tuple.append((title, idx, ratio))
            # sort
            match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
            if not match_tuple:
                print('Oops! No match is found')
                return
            if verbose:
                print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]


        def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
            """
            return top n similar movie recommendations based on user's input movie
            Parameters
            ----------
            model_knn: sklearn model, knn model
            data: movie-user matrix
            mapper: dict, map movie title name to index of the movie in data
            fav_movie: str, name of user input movie
            n_recommendations: int, top n recommendations
            Return
            ------
            list of top n similar movie recommendations
            """
            # fit
            model_knn.fit(data)
            # get input movie index
            print('You have input movie:', fav_movie)
            idx = fuzzy_matching(mapper, fav_movie, verbose=True)
        
            distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
        
            raw_recommends =sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
            # get reverse mapper
            reverse_mapper = {v: k for k, v in mapper.items()}
            # print recommendations 
            lol =[]
            for i, (idx, dist) in enumerate(raw_recommends):
                lol.append('{0}: {1}'.format(i+1, reverse_mapper[idx]))

            return lol

        name = make_recommendation(
            model_knn=model_knn,
            data=movie_user_mat_sparse,
            fav_movie=my_favorite,
            mapper=movie_to_idx,
            n_recommendations=10)
        
        context={
            'ml' : name,
        }
        return render(request, "app1/collabres.html", context)

    else:
        return HttpResponse("PLEASE TYPE A MOVIE NAME")    

def see(request):
    return render(request, "app1/visual.html")