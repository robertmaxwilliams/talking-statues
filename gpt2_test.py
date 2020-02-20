import gpt_2_simple as gpt2
import os
import requests

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# # Downloading model
# model_name = "124M"
# if not os.path.isdir(os.path.join("models", model_name)):
#     print(f"Downloading {model_name} model...")
#     gpt2.download_gpt2(model_name=model_name)  

# # Get user input
# prompt = "Once apon a time there was a "
# prefix = input(f'\n{prompt}... : ')
# if not prefix:
#     prefix = "frog named Michael J. Fox"
# prefix = prompt + prefix

# # Start session and generate text
# sess = gpt2.start_tf_sess()
# gpt2.load_gpt2(sess)
# output = gpt2.generate(sess, prefix=prefix, include_prefix=False, return_as_list=True)[0]
# print(f'\n\n{bcolors.OKGREEN}{prefix} {bcolors.ENDC}{output}\n')

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

def strip_in(text, inside):
    return text[text.find(inside)+len(inside):]

def generate_text(prefix):
    prefix = prefix[:-100]
    output = gpt2.generate(sess, prefix=prefix, include_prefix=False, return_as_list=True, length=100, batch_size=5,
    nsamples=5)
    # print(f'\n\n{bcolors.OKGREEN}{prefix} {bcolors.ENDC}{output}\n')
    output = [x.lstrip(prefix) for x in output]
    return output