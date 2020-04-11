import gpt_2_simple as gpt2
import os
import requests
from random import randint as dice
import time
import markovify
from OpenSSL import SSL

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
if (sess != None):
    gpt2.reset_session(sess)

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, multi_gpu=False)
# gpt2.load_gpt2(sess, multi_gpu=True)

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

app = Flask(__name__)
# run_with_ngrok(app) 

# build text model
with open("../static/corpus.txt", encoding="utf8") as f:
    corpus = f.read()

text_model = markovify.Text(corpus)


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
    words = text_model.word_split(text)
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

@app.route("/webhook", methods=["POST"])
def webhook():
    wh_start= time.time()
    print("WEbhhook called!")
    # body = request.body.getvalue().decode('utf-8')
    # print_red(body)
    # body = json.loads(body)
    body = request.get_json()
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
    generated_story = generate_from_text_model(story, 1, num_samples=1)
    print(generated_story)
    # story += 'THE PLOT!'
    json_text = f'{{"fulfillmentText": "{story}"}}'
    wh_end= time.time()
    print(f"Time to respond: {wh_end - wh_start}")
    return json_text

# context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
# context.use_privatekey_file('/etc/letsencrypt/live/sleepstorymachine.xyz/privkey.pem')
# context.use_certificate_file('/etc/letsencrypt/live/sleepstorymachine.xyz/fullchain.pem')

# context = ('/etc/letsencrypt/live/sleepstorymachine.xyz/fullchain.pem', '/etc/letsencrypt/live/sleepstorymachine.xyz/privkey.pem')
context = ('cert.crt', 'priv.key')

app.run(host="0.0.0.0", port=5000, ssl_context=context)
