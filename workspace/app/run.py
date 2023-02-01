import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    #tokenize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    sumTemp = df.drop(['id', 'message','genre'], axis = 1)  
   
    # createe new data base with sums of the category data
    totals = pd.DataFrame(columns=['Category', 'Total'])
    totals['Category'] = sumTemp.sum(axis = 0).keys()
    totals['Total'] = sumTemp.sum(axis = 0).values
    
    
    totals  = totals.sort_values("Total", ascending=False)
    print(totals[:5])
    topTot = totals[1:6]
    #print(sums.sort_values("count"))
    genre_counts = df.groupby('genre').count()['message']

    genre_names = list(genre_counts.index)

   
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    #creates graph2 for display in the webapp
    graphs2 = [
        {
            'data': [
                Bar(
                    x=topTot["Category"],
                    y=topTot["Total"]
                )
            ],

            'layout': {
                'title': 'Top Categories',
                'yaxis': {
                    'title': "Total"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ] 


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)


    ids2 = ["graph2-{}".format(i) for i, _ in enumerate(graphs2)]
    graphJSON2 = json.dumps(graphs2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, ids2 = ids2, graphJSON2 = graphJSON2)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()