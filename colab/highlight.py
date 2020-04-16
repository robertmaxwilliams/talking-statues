import os
import json

import tensorflow as tf
import numpy as np

from gpt_2_simple.src.sample import *
from gpt_2_simple.src import model, sample, encoder
import gpt_2_simple as gpt2

def sample_sequence(*, hparams, length, start_token=None,
                    batch_size=None, context=None, temperature=1,
                    top_k=0, top_p=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
    context_head = context[:,0:1]
    context_tail = context[:,1:]

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens,
                                past=past, reuse=tf.compat.v1.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(
            hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.compat.v1.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context_head[:, :-1])

        def body(past, prev, context_head, context_tail, all_logits):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.cast(temperature, tf.float32)
            only_logits = logits
            #if top_p > 0.0:
            #    logits = top_p_logits(logits, p=top_p)
            #else:
            #    logits = top_k_logits(logits, k=top_k)
            #samples = tf.random.categorical(
            #    logits, num_samples=1, dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                context_tail[:,0],
                tf.concat([context_head, context_tail[:,0:1]], axis=1),
                context_tail[:,1:],
                tf.concat([all_logits, tf.expand_dims(only_logits, 1)], axis=1)
            ]

        def cond(*args):
            return True

        past, prev, tokens, context_tail, all_logits = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context_head,
                context_tail,
                tf.ones([batch_size, 0, 50257])
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(
                    hparams=hparams, batch_size=batch_size)),#past?
                tf.TensorShape([batch_size]), #prev?
                tf.TensorShape([batch_size, None]),#context head
                tf.TensorShape([batch_size, None]),#context tail
                tf.TensorShape([batch_size, None, 50257]), #all logits
            ],
            back_prop=False,
        )

        return past, prev, tokens, all_logits

def get_text_rankings(sess, string):
    '''takes string, returns pairs of (string, int) 
    where the string is a part of the input and the int it it's gpt2 ranking with
    0 being most likely'''
    prefix = string
    batch_size=1

    #sess = gpt2.start_tf_sess()
    #gpt2.load_gpt2(sess, multi_gpu=False)
    checkpoint_path = os.path.join('checkpoint', 'run1')
    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    context_tokens = [50256] + enc.encode(prefix)
    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)

    past, prev, output, all_logits = sample_sequence(hparams=hparams, length=len(context_tokens)-1, start_token=None, batch_size=batch_size, context=context, temperature=1, top_k=0, top_p=0.0)
    output = output[:, 1:]

    pas, out, alt = sess.run([past, output, all_logits],
                  feed_dict={context: batch_size * [context_tokens]})

    text = enc.decode(out[0])
    print(text)
    print('generated tokens shape: ', out[0].shape)
    print('pas shape: ', pas.shape)

    def find_ranking(arr, index):
        "finds the postions of the element at index if the array was sorted decreasing"
        return (arr > arr[index]).sum()

    string_rank_pairs = []
    for i, token in enumerate(context_tokens[1:]):
        logs = alt[0][i]
        token_string = enc.decode([token])
        token_ranking = find_ranking(logs, token)
        string_rank_pairs.append((token_string, token_ranking))

    #gpt2.reset_session(sess)

    return string_rank_pairs
