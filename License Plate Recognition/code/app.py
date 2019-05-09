from flask import Flask, request, jsonify, render_template
from flask_pymongo import PyMongo
from bson.json_util import dumps
from nocache import nocache
import re
import os
import glob
import infer

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://127.0.0.1:27017/hackathon-db"
mongo = PyMongo(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

# if app.config["DEBUG"]:
#     @app.after_request
#     def after_request(response):
#         response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
#         response.headers["Expires"] = 0
#         response.headers["Pragma"] = "no-cache"
#         return response


@app.route('/index', methods=['GET'])
def index_page():
    return render_template('index.html')


@app.route('/upload')
@nocache
def upload():
    files = glob.glob("static/Process/*.jpg")
    for file in files:
        os.remove(file)

    return render_template('upload.html')


@app.route('/uploaded', methods=['POST'])
@nocache
def uploaded():
    # get file path
    target = os.path.join(APP_ROOT, 'static/')
    # print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("inputfile"):
        # print(file)
        filename = file.filename
        print(filename)
        destination = "\\".join([target, filename])
        # print(destination)
        file.save(destination)

    # execute image-processing python scripts, using the file-path, which will generate the registration number
    global generated_registration
    generated_registration = infer.KNN(destination)
    print('The registration number is:' + generated_registration)

    # return a html render-template that will be used to show a single user based on the registration number that has been generated from the script
    # mongodb find operation will be executed

    return render_template("uploaded.html",
                           file_name=filename,
                           registration=str(generated_registration),
                           resized="Process/2 - Resized.jpg",
                           threshold="Process/3 - Threshold.jpg",
                           sharpen="Process/5 - Sharpen.jpg",
                           identified="Process/identified_characters.jpg")


@app.route('/users/one', methods=['POST'])
def upload_image():
    # search for the user in mongo based on the registration number
    users = mongo.db.users
    result = users.find({'registrationNumber': generated_registration})

    # return a html render-template that will be used to show a single user based on the registration number that has been generated from the script
    # mongodb find operation will be executed
    return render_template("get_user.html", result=result, registration=str(generated_registration))


@app.route('/users/all', methods=['GET'])
def find_users():
    users = mongo.db.users
    result = users.find()

    return render_template('get_all_users.html', result=result)


@app.route('/add')
def add():
    return render_template("add_user.html")


@app.route('/users/add', methods=['POST'])
def add_a_user():
    registration_number = request.form['registrationNumber']
    first_name = request.form['firstName']
    last_name = request.form['lastName']
    email = request.form['email']
    phone_number = request.form['phoneNumber']
    car_model = request.form['carModel']
    car_colour = request.form['carColour']
    car_year = request.form['carYear']

    users = mongo.db.users
    users.insert({'registrationNumber': registration_number, 'firstName': first_name, 'lastName': last_name,
                  'email': email, 'phoneNumber': phone_number, 'carModel': car_model, 'carColour': car_colour, 'carYear': car_year})

    return add()


@app.route('/users', methods=['POST'])
def add_user():
    users = mongo.db.users
    data = request.get_json()

    registration_number = data['registrationNumber']
    first_name = data['firstName']
    last_name = data['lastName']
    users.insert({'registrationNumber': registration_number,
                  'firstName': first_name, 'lastName': last_name})

    return 'Added user'


@app.route('/users/<registration_number>', methods=['GET'])
def find_user(registration_number):
    users = mongo.db.users
    result = users.find_one(
        {'registrationNumber': re.compile(registration_number, re.IGNORECASE)})

    return dumps(result)


@app.route('/users/<registration_number>', methods=['DELETE'])
def remove_user(registration_number):
    users = mongo.db.users

    users.delete_one({'registrationNumber': re.compile(
        registration_number, re.IGNORECASE)})

    return 'User removed'


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)  # run as localhost:8000
