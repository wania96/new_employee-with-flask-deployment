import numpy as np
import pickle
from flask import Flask,request,jsonify,render_template


app = Flask(__name__)
# convert load the pickle module 
#f = open(r"model.pickle","rb")
#text = f.read()
#model = pickle.load(text)
model = pickle.load(open(r'model.pickle', 'rb'))

@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    # this will take the input from the form
    int_features = [float(i) for i in request.form.values()]
    # int_features = [6.5,5.4,3.2,1.8]
    final_features = [np.array(int_features)]
    # [[6.5,5.4,3.2,1.8],]
    prediction = model.predict(final_features)
    # prediction :   [0]         [1]              [2]
    # prediction[0]:  0           1                 2       3
    #              excellent      outstanding       good    
    ref = ["excellent","outstanding","good","low"]
   
    #        0           1           2
    output = ref[prediction[0]]

    return render_template('output.html', prediction_text='prediction{}'.format(output.upper()))

@app.route('/results',methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
   # app.run(debug=)
    app.run(debug=True, use_reloader=False)