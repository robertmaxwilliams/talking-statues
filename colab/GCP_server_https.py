from flask import Flask, render_template, send_from_directory, request
import gpt_2_simple as gpt2
import json
import os
import random
from random import randint as dice
import requests
import threading
import time
import colorsys

from highlight import get_text_rankings

app = Flask(__name__)

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, multi_gpu=False)

#! Classes
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LongRunner:
    working = False
    ready = False
    story = None
    asked = False
    
    def start(self, query_text):
        run_start = time.time()

        # print(f'LongerRunner started: query = {query_text[100:]}')
        print(f'LongerRunner started: query = {query_text}')

        self.working = True
        self.ready = False
        self.asked = False

        # time.sleep(20)
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
        
        story = generate_text(prefix, 100, 1)[0]
        # story = prefix[prefix.rfind('\'')+1:] + story

        print('Returning a story:', story)

        self.story = story


#! Functions


def generate_text(prefix, length, num_samples):
    
    # prefix = prefix[:length]
    print(f'Generate: prefix = {prefix}')
    # print(len(prefix))
    output = gpt2.generate(
        sess,
        prefix=prefix,
#         include_prefix=False,
        include_prefix=True,
        return_as_list=True,
        length=length,
        nsamples=num_samples,
        # batch_size=num_samples,
    )
    output = [x[len(prefix):] for x in output]


    return output


def generate_from_text_model(text, length=100, num_samples=1):
    gems = generate_text(text, length, num_samples)
    ret = ''
    try:
        for i in range(num_samples):
            ret += f'<div class=\'predictionBox\'> <p><pre>{gems[i]}</pre></p> </div>'
    except Exception as e:
        print(f'FEEE\n{e}\nEEEEF')
        return 'sorry, model failure: ' + str(e)
    return ret


def random_color():
    return '{:06x}'.format(dice(0, 0xFFFFFF))

def color_from_rank(rank):
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
    string_rank_pairs = get_text_rankings(sess, text)
    ret = ''
    for word, rank in string_rank_pairs:
        ret += f'<span style=\'background-color: {color_from_rank(rank)}\'>{word}</span>'
    return ret


#! Routes

''' Default Route '''
@app.route('/')
def index():
    return send_from_directory('../static/', 'foo.html')


@app.route('/static/<filename>')
def server_static(filename):
    return send_from_directory('../static/', filename)


@app.route('/generate', methods=['POST'])
def generate():
    generate_start = time.time()
    
    text = request.form['text']
    print(f'Generate POST: prefix = {text}')
    stories = generate_from_text_model(text, 100, num_samples=5)
    
    generate_end = time.time()
    print(f'Time to respond: {generate_end - generate_start}')
    
    return stories


@app.route('/highlight', methods=['POST'])
def highlight():
    time.sleep(0.5)
    text = request.form['text']
    print(f'Highlight: text = {text}')
    return '<p>' + colorize(text) + '</p>'

def print_red(text): 
    print('\033[91m {}\033[00m'.format(text))


# exists so calling from thread is easilier
def long_runner_helper(query_text):
    print(f'Helper: query (len={len(query_text)}) = {query_text}')
    long_runner.start(query_text)

@app.route('/webhook', methods=['POST'])
def webhook():
    hook_start = time.time()
    print('Webhhook called')

    # Get relevant message data
    body = request.get_json()
    query_text = body['queryResult']['queryText']

    # print(f'Webhook: query = {query_text[100:]}')
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



# Restart session if running this cell again
# sess = None
# if (sess != None):
#    gpt2.reset_session(sess)

#sess = gpt2.start_tf_sess()
#gpt2.load_gpt2(sess, multi_gpu=False)

long_runner = LongRunner()

# context = ('/etc/letsencrypt/live/sleepstorymachine.xyz/fullchain.pem', '/etc/letsencrypt/live/sleepstorymachine.xyz/privkey.pem')
context = ('cert.crt', 'priv.key')

app.run(host='0.0.0.0', port=5000, ssl_context=context)

