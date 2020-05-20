import json
import os
import random
from random import randint as dice
import requests
import threading
import time
import colorsys

import tensorflow as tf
import gpt_2_simple as gpt2
from flask import Flask, render_template, send_from_directory, request

from highlight import get_text_rankings

app = Flask(__name__)


# ============
# Classes
# ============
class LongRunner:
    '''Bascially a poor work queue for only 1 task'''
    working = False
    ready = False
    story = None
    asked = False

    def start(self, query_text):
        run_start = time.time()

        print(f'LongerRunner started: query = {query_text}')

        self.working = True
        self.ready = False
        self.asked = False

        self.generate_story_from_query_text(query_text)

        self.working = False
        self.ready = True

        run_end = time.time()
        print(f'LongRunner done!, took {run_end - run_start} to finish')

    def get_result(self):
        if self.ready:
            self.ready = False
            return (self.story, False)
        elif not self.asked:
            self.asked = True
            return ('Okay, let me think of a story', True)
        else:
            return ('I\'m still thinking...', True)

    def generate_story_from_query_text(self, query_text):
        prefix = random.choice(story_prefix_templates).format(query_text)
        print(f'Rand: prefix = {prefix}')

        story = generate_text_threadsafe(prefix, 100, 1)[0]
        story = prefix[prefix.rfind('"')+1:] + story

        print('Returning a story:', story)

        self.story = story


def long_runner_helper(query_text):
    '''helper for LongRunner so it can be called as a function'''
    print(f'Helper: query (len={len(query_text)}) = {query_text}')
    long_runner.start(query_text)

#! Functions
def generate_text_threadsafe(prefix, length, num_samples):
    '''Utility function to call gpt-2 in a thread'''
    print(f'Generate ts: prefix = {prefix}')
    global sess
    global graph
    with graph.as_default():
        with sess.as_default():
            output = gpt2.generate(
                sess,
                prefix=prefix,
                include_prefix=True,
                return_as_list=True,
                length=length,
                nsamples=num_samples,
                batch_size=num_samples,
            )
            output = [x[len(prefix):] for x in output]
    return output

def generate_text(prefix, length, num_samples):
    '''Utility function to call gpt-2'''
    print(f'Generate: prefix = {prefix}')
    output = gpt2.generate(
        sess,
        prefix=prefix,
        include_prefix=True,
        return_as_list=True,
        length=length,
        nsamples=num_samples,
        batch_size=num_samples,
    )
    output = [x[len(prefix):] for x in output]
    return output


def generate_html_boxes_from_text_model(text, length=100, num_samples=1):
    '''Returns a div for each sample with the text in it'''
    gems = generate_text(text, length, num_samples)
    ret = ''
    try:
        for i in range(num_samples):
            ret += f'<div class=\'predictionBox\'> <p>{gems[i]}</p> </div>'
    except Exception as e:
        print(f'FEEE\n{e}\nEEEEF')
        return 'sorry, model failure: ' + str(e)
    return ret

def color_from_rank(rank):
    '''These are the colors for highlighting text. Take integer 0 to 50,257'''
    mustard = (0xFB, 0xB8, 0x09)
    raspberry = (0x7A, 0x34, 0x58)
    maroon = (0x52, 0x18, 0x2E)
    if rank == 0:
        color, opacity = mustard, 0.1
    elif rank < 10:
        color, opacity = mustard, 0.2
    elif rank < 100:
        color, opacity = mustard, 0.4
    elif rank < 1000:
        color, opacity = raspberry, 0.4
    elif rank < 5000:
        color, opacity = raspberry, 0.6
    elif rank < 10000:
        color, opacity = raspberry, 0.7
    elif rank < 30000:
        color, opacity = maroon, 0.8
    else: # 50257 is the maximum (worst) ranking
        color, opacity = maroon, 0.95
    return f"rgba({color[0]}, {color[1]}, {color[2]}, {opacity})"


def colorize(text):
    '''takes text, tokenizes and rates tokens the returns html with colorized text'''
    string_rank_pairs = get_text_rankings(sess, text)
    ret = ''
    for word, rank in string_rank_pairs:
        ret += f"<span style='background-color: {color_from_rank(rank)}'>{word}</span>"
    return ret

# ============
# Routes
# ============

@app.route('/')
def index():
    '''Default route'''
    return send_from_directory('./static/', 'foo.html')

@app.route('/about')
def about():
    '''Default route'''
    return send_from_directory('./static/', 'about.html')


@app.route('/static/<filename>')
def server_static(filename):
    '''servers files in the static directory'''
    return send_from_directory('./static/', filename)

@app.route('/generate', methods=['POST'])
def generate():
    '''Endpoint for the generate button'''
    generate_start = time.time()

    text = request.form['text']
    print(f'Generate POST: prefix = {text}')
    stories = generate_html_boxes_from_text_model(text, 100, num_samples=3)

    generate_end = time.time()
    print(f'Time to respond: {generate_end - generate_start}')

    return stories

@app.route('/highlight', methods=['POST'])
def highlight():
    '''endpoint for highlighting button'''
    text = request.form['text']
    print(f'Highlight: text = {text}')
    return colorize(text)

@app.route('/webhook', methods=['POST'])
def webhook():
    hook_start = time.time()
    print('Webhhook called')

    # Get relevant message data
    body = request.get_json()
    query_text = body['queryResult']['queryText']

    print(f'Webhook: query (len={len(query_text)}) = {query_text}')

    # Handle welcom page request
    if query_text == 'GOOGLE_ASSISTANT_WELCOME':
        print('Welcome page, returning empty json object')
        return '{}'

    # Start long runner if needed
    if not (long_runner.working or long_runner.ready):
        x = threading.Thread(target=long_runner_helper, args=(query_text,))
        x.start()

    result, expect_response = long_runner.get_result()

    hook_end = time.time()
    print(f'WH Time to respond: {hook_end - hook_start}')

    return json.dumps({
        'fulfillmentText': result,
        'payload': {
            'google': {
                'expectUserResponse': expect_response,
                'richResponse': {
                    'items': [{
                        'simpleResponse': {
                            'textToSpeech': result
                            }
                        }]
                    }
                }
            }
        })


# ============
# Global constants
# ============

story_prefix_templates = [
        (
            'The old man asked what I wanted to hear a story about. '
            'I pushed a coin his way and said "{}". He began his story, "Once upon a time,'
        ),
        (
            '"Grandma, tell me a story!"\n'
            '"Of course dear, what about?"\n'
            '"{}"\n'
            '"Well, once upon a time'
        ),
        (
            'We were all sitting around the campfire, and it was my turn to tell a story.\n'
            '"{}" one of them piped up from the other side of the fire.\n'
            'I had a perfect story, which I told without interruption: "A long, long time ago,'
        )
]

long_runner = LongRunner()

sess = gpt2.start_tf_sess()
graph = tf.get_default_graph()
gpt2.load_gpt2(sess, multi_gpu=False)

# ===========
# Main
# ============
if __name__ == "__main__":
    context = ('cert.crt', 'priv.key')
    app.run(host='0.0.0.0', port=5000, ssl_context=context)
