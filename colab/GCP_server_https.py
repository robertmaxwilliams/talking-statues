#import gpt_2_simple as gpt2
import os
import requests
from random import randint as dice
import random
import time
import json

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

sess = None

# Restart session if running this cell again
'''
if (sess != None):
    gpt2.reset_session(sess)

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, multi_gpu=False)
# gpt2.load_gpt2(sess, multi_gpu=True)
'''
class FakeGenerator:
    words = ["the", "of", "for", "by", "when", "of", "a"]
    def generate(self, *args, prefix="", length=5, n_samples=1, **kwargs):
        gens = []
        for _ in range(n_samples):
            gens.append(prefix + " " + " ".join(random.choices(self.words, k=length)) + ".")
        return gens
gpt2 = FakeGenerator()
print(gpt2.generate(
        sess,
        prefix="foobar",
        include_prefix=True,
        return_as_list=True,
        length=4,
        nsamples=5,
        ))


def generate_text(prefix, length, num_samples):
    prefix = prefix[:length]
    output = gpt2.generate(
        sess,
        prefix=prefix,
#         include_prefix=False,
        include_prefix=True,
        return_as_list=True,
        length=length,
        nsamples=num_samples,
#         batch_size=num_samples,
    )
    output = [x[len(prefix):] for x in output]
    return output


from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, send_from_directory, request
import time

import threading


app = Flask(__name__)
# run_with_ngrok(app) 


# build text model
with open("../static/corpus.txt", encoding="utf8") as f:
    corpus = f.read()


def generate_from_text_model(text, length=100, num_samples=1):
    gems = generate_text(text, length, num_samples)
    ret = ""
    try:
        for i in range(num_samples):
            ret += f"<div class='predictionBox'> <p><pre>{gems[i]}</pre></p> </div>"
    except Exception as e:
        print(f"FEEE\n{e}\nEEEEF")
        return "sorry, model failure: " + str(e)
    return ret


def random_color():
    return "{:06x}".format(dice(0, 0xFFFFFF))


def colorize(text):
    words = text.split(" ")
    ret = ""
    for w in words:
        ret += f"<span style='background-color:#{random_color()}'>{w}</span>"
    return ret


@app.route("/static/<filename>")
def server_static(filename):
    return send_from_directory("../static/", filename)


@app.route("/")
def index():
    return send_from_directory("../static/", "foo.html")


@app.route("/generate", methods=["POST"])
def generate():
    generate_start= time.time()
    
    text = request.form["text"]
    print(f"Generate: prefix = {text}")
    stories = generate_from_text_model(text, 20, num_samples=1)
    
    generate_end = time.time()
    print(f"Time to respond: {generate_end - generate_start}")
    
    return stories


@app.route("/highlight", methods=["POST"])
def highlight():
    time.sleep(0.5)
    text = request.form["text"]
    print(f"Highlight: text = {text}")
    return "<p>" + colorize(text) + "</p>"

def print_red(text): 
    print("\033[91m {}\033[00m".format(text))

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





class LongRunner:
    working = False
    ready = False
    story = None
    asked = False
    def start(self, query_text):
        self.working = True
        self.ready = False
        self.asked = False
        time.sleep(20)
        self.generate_story_from_query_text(query_text)
        self.working = False
        self.ready = True
        print("done!")
    def get_result(self):
        if self.ready:
            self.ready = False
            return (self.story, False)
        elif not self.asked:
            self.asked = True
            return ("Okay, let me think of a story", True)
        else:
            return ("I'm still thinking...", True)
    def generate_story_from_query_text(self, query_text):
        prefix = random.choice(story_prefix_templates).format(query_text)
        story = generate_text(prefix, 5, 1)[0]
        story = prefix[prefix.rfind('"')+1:] + story
        print("Returning a story:", story)
        self.story = story
long_runner = LongRunner()

# exists so calling from thread is easilier
def long_runner_helper(query_text):
    long_runner.start(query_text)

@app.route("/webhook", methods=["POST"])
def webhook():
    print("WEbhhook called!")
    body = request.get_json()
    query_text = body['queryResult']['queryText']
    if query_text == "GOOGLE_ASSISTANT_WELCOME":
        print("Welcome page, returning empty json object")
        return '{}'
    if not (long_runner.working or long_runner.ready):
        x = threading.Thread(target=long_runner_helper, args=(query_text,))
        x.start()
    result, expect_response = long_runner.get_result()
    return json.dumps(
            {"fulfillmentText": result, 
                "payload": {
                    "google": {
                        "expectUserResponse": expect_response,
                        "richResponse": {
                            "items": [
                                {
                                    "simpleResponse": {
                                        "textToSpeech": result
                                        }
                                    }
                                ]
                            }
                        }
                    }
                })

# context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
# context.use_privatekey_file('/etc/letsencrypt/live/sleepstorymachine.xyz/privkey.pem')
# context.use_certificate_file('/etc/letsencrypt/live/sleepstorymachine.xyz/fullchain.pem')

# context = ('/etc/letsencrypt/live/sleepstorymachine.xyz/fullchain.pem', '/etc/letsencrypt/live/sleepstorymachine.xyz/privkey.pem')
##context = ('cert.crt', 'priv.key')

app.run(host="localhost", port=5000)#, ssl_context=context)
#https://sleepstorymachine.xyz:5000/webhook

