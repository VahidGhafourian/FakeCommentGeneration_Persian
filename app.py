from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
app.logger.setLevel('INFO')
api = Api(app)


# @app.route('/')
# def hello():
#     return "Hello world"
class Hello(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(Hello, '/hello')

if __name__ == "__main__":
    app.run(debug=True)