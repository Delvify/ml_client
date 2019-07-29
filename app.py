from flask import Flask, render_template
from flask import request
from bert_serving.client import BertClient
from numpy import dot
from numpy.linalg import norm
from flask_cors import CORS
import json
import os
from flask.json import jsonify
from gevent import monkey
monkey.patch_all()
from gevent import pywsgi
import operator


app = Flask(__name__)
cors = CORS(app, resources={r"/computeSimilarity" : {"origins": "*"}})


global item_sentenceEmbedding


@app.route("/")
def main():
    '''
    Main Function -> this is the point where program starts \n
    :return: Returns nothing
    '''
    global item_sentenceEmbedding
    item_sentenceEmbedding = getSentenceEmbeddingAllItem()
    return render_template("index.html")


@app.route("/about")
def about():
    '''
    To Check if the serving is up and running.
    :return:
    '''
    return "About"


def get_bertClient():
    '''
    BERT client object connects to the Bert server \n
    :return: bert client object
    '''
    bc = BertClient()
    return bc


def getItemDescription():
    # for the shake of example, creating 10 sentences in a list

    item_description = dict()
    item_description[0] = "Terrariums are a popular gift and look fabulous in any room of the house. Lush crassula and rosette succulents take centre stage in this mini fishbowl terrarium. A fantastic way to express your thoughts to your loved ones and perfect for almost any occasion including birthdays or to say congratulations, Tammy will be sure to impress and leave them grinning from ear to ear. We offer same day delivery, we have over 100 years of experience and we guarantee fresh, quality flowers and gifts. Make their day and buy flowers online now!"
    item_description[1] = "A delightful combination of roses, lilies, gerberas, and carnations in varying shades of fuschia, this floral gift will be adored. Expertly handcrafted by a local florist and presented in a white ceramic pot, Berry Delight is ideal for a birthday or special ocassion"
    item_description[2] = "This unique floral gift is brimming with beautiful pinks, soft yellows, and green foliage that is sure to delight. Proteas add a special touch to this bouquet that is presented in a modern glass vase. Treat them to Bliss."
    item_description[3] = "This rainbow of seasonal blooms will brighten their day! Perfect for a birthday or the colour lover in your life, Splice is expertly presented in a glass fishbowl ideal for reuse. Complete their gift with a bright balloon or sparkling wine"
    item_description[4] = "Surprise someone with this colourful gift. Filled to the brim with fresh seasonal fruits and a bottle of premium sparkling wine, they won't be able to resist this lavishly presented arrangement"
    item_description[5] = "Vibrant colours in a mix of textures abound in this bright mixed arrangement. Lilies, gerberas, and roses are amongst the choice blooms in this floral gift, presented in a white box with cheery orange ribbon. Fiesta is perfect for any celebration"
    item_description[6] = "A delightful boutique well suited to any taste, Classique combines pale greens with fresh whites in a range of textures, highlighted by sweetly-scented eucalyptus. Presented in a stylish vase perfect for reuse, they will love this beautiful gesture. This gift is ideal for a birthday, a thank you gift, or to congratulate them."
    item_description[7] = "A sophisticated gift for the lover of natives and wildflowers, this arrangement is a stunning combination of earth tones in striking textures. Expertly handcrafted and presented in a white ceramic vase, this premium arrangement is sure to please"
    item_description[8] = "The unique Citrus features fiery toned blooms highlighted by deep green foliage. Presented in an elegant ceramic container, they will love this surprise. Add balloons or chocolates to make this gift even more special"
    item_description[9] = "Send them get well wishes with this cute gift! A cuddly teddy bear is presented in a cute box with bright coloured helium balloons, the perfect way to lift their spirits"

    return item_description


def read_catalog():
    '''
      Reads the catalog file of items as an input \n
      :param: Input File path \n
      :returns: the dictionary of key (sku) and values (description)
    '''
    item_dictionary = dict()
    # catalog_filePath = "/home/jugs/PycharmProjects/DelvifyFWD/web/resources/DataFeedInterfloraProductFeed.json"
    catalog_filePath = "/home/jugs/PycharmProjects/DelvifyFWD/CatalogPreprocessor/rawJson/DataFeedYOINS.json"
    with open(catalog_filePath, "r") as catalog_file:
        data = json.load(catalog_file)
    for line in data:
        sku = line["SKU"]
        name = line["Name"]
        category = line["Category"]
        description = line["Description"]
        if len(description) < 50:
            pass
        item_dictionary[sku] = category + ". " + name + ". " + description
    return item_dictionary


def connection():
    '''
    Pass the sql connection string and login/authorization credential \n
    :return: Returns the connection Object
    '''
    conn = "mysql connection"
    return conn

@app.before_first_request
def getSentenceEmbeddingAllItem():
    '''
     This function generates the sentence embedding fot a given list of string \n
    :return: dictionary of key as sku and value as sentence_embedding
    '''
    app.logger.info("Running Application Once!")
    bc = get_bertClient()
    items_dict = read_catalog()
    description_embeddings = bc.encode(list(items_dict.values()))
    i = 0
    for key, value in items_dict.items():
        items_dict[key] = description_embeddings[i]
        i = i + 1
    return items_dict


def getSentenceEmbeddingQuery():
    return str


def cos_sim(a, b):
    '''
    Takes in two numpy array as it's parameter,
    and returns the cosine similarity of two vector(numpy)
    :param a: numpy array
    :param b: numpy array
    :return: cosine similarity score (scalar quantity) with a given equation
    '''
    return dot(a, b)/(norm(a)*norm(b))


def computeSimilarityScore(queryEmbededing):
    '''
    Takes the input parameters as the sentence embedding that was returned from the
    bert_client which was hosted from the server. And, uses the globally declared
    variable (list of sentence embedding). Then generates the list of scores by
    computing the similarity between user input and corpus.
    :param queryEmbededing:
    :return: list of score with user query and data corpus.
    '''
    global item_sentenceEmbedding
    scores = dict()
    # queryEmbededing = np.transpose(queryEmbededing)
    for key, value in item_sentenceEmbedding.items():
        score_val = cos_sim(queryEmbededing[0], value)
        scores[key] = score_val
    return dict(sorted(scores.items(), reverse=True, key=operator.itemgetter(1)))


def get_top_n():
    # return top n similar items as that of input query
    return


def getQueryEmbedding(query):
    '''
    Takes in the string/user response as an input, process with the bert_client,
    to generate the sentence embedding of the given input
    :param query: String value (input sentence)
    :return: sentence embedding tensor
    '''
    bc = get_bertClient()
    query_embedding = bc.encode(query)
    return query_embedding

@app.route('/computeSimilarity')   # , methods=['GET', 'POST'])
def computeSimilarity():
    score = dict()
    # usertext = request.get_data()
    usertext = request.args.get('text')
    # query_txt = usertext.decode('utf-8')
    query_lst = list()
    query_lst.append(usertext)
    query_embed = getQueryEmbedding(query_lst)
    score = computeSimilarityScore(query_embed)
    topn_skus = get_topn_items(score)
    # score = [423, 456, 766, 123, 345, 343, 867, 642, 702, 703, 771, 775, 780]
    return jsonify({'skus':topn_skus})         # str(usertext) + "Delvify"


def get_topn_items(item_dict, n =30):
    skus = list()
    i = 0
    for key, value in item_dict.items():
        if i <= n:
            skus.append(key)
            i = i + 1
    return skus

if __name__ == "__main__":
    global item_sentenceEmbedding
    try:
        server = pywsgi.WSGIServer(('0.0.0.0', 5001), app)
        server.serve_forever("curl http://0.0.0.0:5000")
        if True:
            os.system('curl http://0.0.0.0:5001')
    except KeyboardInterrupt:
        print("closed")
    # app.run(debug=True, host="0.0.0.0")


#  curl http://13.67.88.182:8085/computeSimilarity?text=red%20wedding%20flower

'''
Runing in the production do not use: python app.py
Instead Use: gunicorn --bind=0.0.0.0:5000  --timeout 600 app:app
            OR
server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
server.serve_forever()
            AND <--- For debugging & Testing --->
app.run(debug=True, host="0.0.0.0")
'''


'''
First Run the Model-server as:
  $ bert-serving-start -model_dir /location/to/bert_pretrained_model/uncased_L-12_H-768_A-12/1/ -num_workers 4
The run a host-server
  $
And, 
  $ curl http://ipaddress:port
And, (optionally)
  $ curl http://ip_address:port/computeSimilarity?text=red%20wedding%20flower
'''
