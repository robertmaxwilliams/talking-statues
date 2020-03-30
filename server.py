from bottle import static_file
from bottle import Bottle, run, request
from random import randint as dice
import time
import json
import random

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

# TODO make sure these tokenize well - should the end with spaces and or punctuation?
story_prefix_templates = [
('The old man asked what I wanted to hear a story about. '
'I pushed a coin his way and said "{}". He began his story, "Once upon a time,'),
'''"Grandma, tell me a story!"
"Of course dear, what about?"
"{}"
"Well, once upon a time''',
'''We were all sitting around the campfire, and it was my turn to tell a story.
"{}" one of them piped up from the other side of the fire.
I had a perfect story, which I told without interruption: "A long, long time ago,'''
]


@app.post("/webhook")
def webhook():
    print("WEbhhook called!")
    print_red(request.body.getvalue().decode('utf-8'))
    body = json.loads(request.body.getvalue().decode('utf-8'))
    query_text = body["queryResult"]["queryText"]
    if query_text == "GOOGLE_ASSISTANT_WELCOME":
        print("Welcome page, returning empty json object")
        return '{}'
    else:
        prefix = random.choice(story_prefix_templates).format(query_text)
        story = gpt2_test.generate_text(prefix)[0] # TODO only do batch of one, is it still a list?
        # TODO test the following to make sure it is "Once upun a time" or whatever + generated text
        story = prefix[prefix.rfind('"')+1:] + story
        print("Returning a story:", story)
        return json.dumps({"fulfillmentText": story})

run(app, host="localhost", port=8080)
