from bottle import static_file
from bottle import Bottle, run, request
from random import randint as dice
import time
app = Bottle()


# build text model
import markovify
with open("corpus.txt") as f:
    corpus = f.read()
text_model = markovify.Text(corpus)



def generate_from_text_model(text):
    ret = ""
    try:
        for _ in range(5):
            ret += "<div class='predictionBox'> <p>" + text_model.make_sentence_with_start(
                    text_model.word_join(
                        [x.replace('.','') for x in text_model.word_split(text) if x][-2:]),
                       string=False) + "</p> </div>"
    except Exception as e:
        print("FEEE")
        print(e)
        print("EEEEF")
        return "sorry, model failure: " + str(e)
    return ret


def random_color():
    return "{:06x}".format(dice(0, 0xffffff))

def colorize(text):
    words = text_model.word_split(text)
    ret = ""
    for w in words:
        ret += "<span style='background-color:#" + random_color()\
               + "'>" + w  + "</span> "
    return ret

@app.route('/hello')
def hello():
    return "Hello World!"


@app.route('/static/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/')


@app.route('/')
def index():
    return static_file('foo.html', root='./')


@app.post('/generate')
def generate():
    time.sleep(1)
    text = request.forms.get('text')
    return generate_from_text_model(text)

@app.post('/highlight')
def highlight():
    time.sleep(0.5)
    text = request.forms.get('text')
    return "<p>" + colorize(text) + "</p>"

run(app, host='localhost', port=8080)
