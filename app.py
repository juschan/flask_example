import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

app = Flask('myApp')
# load model here!!!
model_that_was_pickled = pickle.load(open('../assets/model.p', 'rb'))

# route 1: hello world
@app.route('/')
def home():
    # return a simple string
    return 'Hello, world - GA!!!!'

# route 2: return a 'web page'
@app.route('/hc_page')
def hc_page():
    # return some hard-coded html
    return '<html><body><h1>This is a hard coded page!</h1><p>Here is some hard-coded content. Isn\'t it pretty?</p></body></html>'

# route 3: return some data
@app.route('/hc_page.json')
def json():
    # create some data to return as json
    best_stuff = {
        'coast': 'west',
        'sandwich': 'el paseo',
        'movie': 'toy story'
    }
    
    # use flask's jsonify function to return the data as well as a 200 status code
    return jsonify(best_stuff), 200


# route 4: show a form to the user
@app.route('/form')
def form():
    # use flask's render_template function to display an html page
    return render_template('form.html')


# route 5: accept the form submission and do something fancy with it
@app.route('/submit')
def make_predictions():
    # load in the form data from the incoming request
    user_input = request.args

    # manipulate data into a format that we pass to our model
    data = [
        int(user_input['OverallQual']),
        int(user_input['FullBath']),
        int(user_input['GarageArea']),
        int(user_input['LotArea'])
    ]
    
    # do the prediction based on user inputs 
    data2 = np.array(data)
    data3 = np.reshape(data, (1, -1))
    price = model_that_was_pickled.predict(data3)[0]
    
    # if you're desperate, you can always calculate directly from coefficients
    price = 32968.7974 * data[0] + 17719.2556 * data[1] + 82.6078 * data[2] + 1.8704 * data[3] 
    
    return jsonify({'saleprice' :  price})
    #return "<html><body><p>The house price is predicted to be $" + price "</p></body></html>"

if __name__ == '__main__':
    app.run(debug=True)
    








