import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_places(placesList, placesData):
    rating_column = 'Aggregate rating'
    votes_column = 'Votes'
    place_name_column = 'Restaurant Name'
    category_column = 'Cuisines'
    id_column = 'Restaurant ID'

    features = [category_column]
    C = placesData[rating_column].mean()
    m = placesData[votes_column].quantile(0.60)

    def get_list(x):
        names = []
        for text in x.split(','):
            names.append(text.lstrip())
        return names

    def clean_data(x):
        if isinstance(x, list):
            return [i.replace(" ", "") for i in x]
        else:
            if isinstance(x, str):
                return x.replace(" ", "")
            else:
                return ''

    def weighted_rating(x, m=m, C=C):
        v = x[votes_column]
        R = x[rating_column]
        return (v/(v+m) * R) + (m/(m+v) * C)

    def create_soup(x):
        return ' '.join(x[category_column])

    def get_recommendations(title, cosine_sim):
        # Get the index of the place that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all places with that place
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the places based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        filtered_sim_scores = [element for element in sim_scores if element[0] != idx]
        # Get the place indices
        places_indices = [i[0] for i in filtered_sim_scores if i[1]>0]

        if(len(places_indices) > 0):
            return placesData.iloc[places_indices]
        else:
            return []

    for feature in features:
        placesData[feature] = placesData[feature].apply(get_list)

    for feature in features:
        placesData[feature] = placesData[feature].apply(clean_data)

    indices = pd.Series(placesData.index, index=placesData[id_column]).drop_duplicates()

    placesData['soup'] = placesData.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(placesData['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    placesData = placesData.reset_index()
    indices = pd.Series(placesData.index, index=placesData[id_column])
    #Combining list of places
    final_list = pd.DataFrame()
    for element in placesList:
        object1 = get_recommendations(element, cosine_sim2)
        if(len(object1) > 0):
            final_list = pd.concat([final_list,object1])

    if(len(final_list) > 0):
        final_list.drop_duplicates(subset = id_column, 
                     keep = 'first', inplace = True) 

        #Filtering by score
        filtered_list = final_list.copy().loc[final_list[votes_column] >= m]
        filtered_list['score'] = filtered_list.apply(weighted_rating, axis=1)
        filtered_list = filtered_list.sort_values('score', ascending=False)
        filtered_list = filtered_list.drop(['score'], axis = 1)

        indices = []
        for element in filtered_list['index']:
            indices.append(element)
    
        remaining_indices = []
        for element in placesData['index']:
            if element not in indices:
                remaining_indices.append(element)
        places_returned_list = placesData.iloc[remaining_indices]

        final_returned_list = pd.concat([filtered_list, places_returned_list])
        final_returned_list = final_returned_list.drop(['index','soup'], axis = 1)
        # print(final_returned_list)
        return final_returned_list
    else:
        return []

# list_of_places = [18384227, 611]
# df = pd.read_csv('./zomato.csv', low_memory=False)
# print(df)
# recommend_places(list_of_places, placesData)
