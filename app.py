from flask import Flask,render_template,request,redirect,jsonify,url_for


app = Flask(__name__)


#flask app routing
@app.route("/", methods=["GET"])

def welcome():
    
    return "<h1>Welcome to Plant Disease Detection</h1>"


@app.route("/index", methods=["GET"])

def index():
    
    return "Welcome to Home Page"

#variable rule
@app.route("/success/<int:accuracy>")
def success(accuracy):
    
    return "The accuracy of that plant is: "+ str(accuracy)


@app.route("/fail")
def fail():
    
    return "Fails to detect the plant "



@app.route('/form',methods=["GET","POST"])

def form():
    
    if request.method=="GET":
        return render_template('form.html')
    else:
        
        acc = float(request.form['advice'])


    return render_template('form.html',accuracy = acc)


@app.route('/api', methods=['POST'])

def advice():
    data = request.get_json()
    
    a_advice = dict(data)['any key']
    b_advice = dict(data)['any key']
    c_advice = dict(data)['any key']
    
    return jsonify(a_advice)

if __name__ == "__main__":
    
    app.run(debug=True)
