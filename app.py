from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from Modules.GenerateComment import CommentGenerator
import pandas as pd

app = Flask(__name__)
app.logger.setLevel('INFO')
api = Api(app)
generator = CommentGenerator()

generator_parser = reqparse.RequestParser()
generator_parser.add_argument('product_title', type=str, help="product_title not sent!")

mainpage_parser = reqparse.RequestParser()


# @app.route('/')
# def hello():
#     return "Hello world"
class DeepCommentFake(Resource):
    # def get(self):
    #     raw , better = generator.generate()
    #     return {'fake Comment': better}
    # @app.route('/predict', methods=['POST'])
    def post(self):
        # print(app.request.values.get('product_title'))
        params = generator_parser.parse_args()
        product_title = params["product_title"]
        raw, better = generator.generate_by_product_title(product_title)
        return jsonify(fake_comment=better)


class Main(Resource):
    # def get(self):
    #     raw , better = generator.generate()
    #     return {'fake Comment': better}
    def post(self):
        self.data = pd.read_csv('Dataset/digi_clean_uni_fullclean_2label_balanced_v1.3_for_noise_correction.csv')
        return jsonify(product_titles=list(set(self.data.product_title.values.tolist())))


api.add_resource(DeepCommentFake, '/predict')
api.add_resource(Main, '/')

if __name__ == "__main__":
    app.run(debug=True)
