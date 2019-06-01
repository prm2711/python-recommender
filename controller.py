from flask import Flask, request
from flask_restful import Resource, Api
from recommender import recommend_places
import pandas as pd
import json

app = Flask(__name__)
api = Api(app)

class Recommendations(Resource):
    def post(self):
        list_of_places = []
        data = request.json['places']
        df = pd.DataFrame.from_dict(data, orient='columns')
        placesUser = request.json['userPlaces']
        df1 = pd.DataFrame.from_dict(placesUser, orient='columns')
        
        for element in df1['placeID']:
            list_of_places.append(element)

        response = recommend_places(list_of_places, df)
        out = response.to_json(orient='records')
        jsonResponse = json.loads(out)
        return jsonResponse 

api.add_resource(Recommendations, '/recommendations')

if __name__ == '__main__':
     app.run(host='192.168.100.155',port='5002')
