from flask import Flask
from flask_restful import Resource, Api

app = Flask("cell_order_API")
api = Api(app)

class sampleRoute(Resource):

    #Get method
    def get(self):
        return "hello"

api.add_resource(sampleRoute,"/")

if __name__ == "__main__":
    app.run()