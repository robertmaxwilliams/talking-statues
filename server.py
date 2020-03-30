from bottle import static_file
from bottle import Bottle, run, request
from random import randint as dice
import time
import json

import gpt2_test

app = Bottle()


def generate_from_text_model(text):
    gems = gpt2_test.generate_text(text)
    ret = ""
    try:
        for i in range(5):
            ret += f"<div class='predictionBox'> <p><pre>{gems[i]}</pre></p> </div>"
    except Exception as e:
        print(f"FEEE\n{e}\nEEEEF")
        return "sorry, model failure: " + str(e)
    return ret


def random_color():
    return "{:06x}".format(dice(0, 0xFFFFFF))


def colorize(text):
    words = text_model.word_split(text)
    ret = ""
    for w in words:
        ret += f"<span style='background-color:#{random_color()}'>{w}</span>"
    return ret


@app.route("/hello")
def hello():
    return "Hello World!"


@app.route("/static/<filename>")
def server_static(filename):
    return static_file(filename, root="./static/")


@app.route("/")
def index():
    return static_file("foo.html", root="./")


@app.post("/generate")
def generate():
    time.sleep(1)
    text = request.forms.get("text")
    return generate_from_text_model(text)


@app.post("/highlight")
def highlight():
    time.sleep(0.5)
    text = request.forms.get("text")
    return "<p>" + colorize(text) + "</p>"

def print_red(text): 
    print("\033[91m {}\033[00m".format(text))

@app.post("/webhook")
def webhook():
    print("WEbhhook called!")
    print_red(request.body.getvalue().decode('utf-8'))
    body = json.loads(request.body.getvalue().decode('utf-8'))
    if body["queryResult"]["queryText"] == "GOOGLE_ASSISTANT_WELCOME":
        print("Welcome page, returning empty json object")
        return '{}'
    parameters = body['queryResult']['parameters']
    story = 'Let me tell you a story. '
    if parameters['subject_thing'] != '':
        story += f"There was a {parameters['subject_thing']}. "
    if parameters['subject_place'] != '':
        story += f"This took place at {parameters['subject_place']}. "
    if parameters['subject_event'] != '':
        story += f"It was a great day, the day of a {parameters['subject_event']}. "
    story += 'THE PLOT!'
    json_text = f'{{"fulfillmentText": "{story}"}}'
    return json_text

run(app, host="localhost", port=8080)
