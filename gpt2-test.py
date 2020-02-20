import gpt_2_simple as gpt2
import os
import requests

#model_name = "1558M"
model_name = "774M"
# model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "shakespeare.txt"
if not os.path.isfile(file_name):
	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	data = requests.get(url)
	
	with open(file_name, 'w') as f:
		f.write(data.text)
    

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
#               batch_size=1,
              multi_gpu=True,
              steps=10)   # steps is max number of training steps

print(dir(gpt2))
print(gpt2.generate(sess, temperature=0.8))
