from flask import Flask
from flask_restful import Resource, Api
from Modules.GenerateComment import CommentGenerator

app = Flask(__name__)
app.logger.setLevel('INFO')
api = Api(app)
generator = CommentGenerator()


# @app.route('/')
# def hello():
#     return "Hello world"
class Hello(Resource):
    def get(self):
        return {'fake Comment': ' '.join(generator.generate())}


api.add_resource(Hello, '/')

if __name__ == "__main__":
    app.run(debug=True)
