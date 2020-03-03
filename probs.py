
import matplotlib.pyplot as plt
from gpt_2_simple import *
from gpt_2_simple.src import sample

plt.xkcd()

def my_sample_sequence(*, hparams, length, start_token=None,
                    batch_size=None, context=None, temperature=1,
                    top_k=0, top_p=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

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
        context_output = step(hparams, context[:, :-1])
        print(f'!?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {context.shape}')

        def body(past, prev, output, logis):
        # def body(past, prev, output):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.cast(temperature, tf.float32)
            if top_p > 0.0:
                logits = sample.top_p_logits(logits, p=top_p)
            else:
                logits = sample.top_k_logits(logits, k=top_k)
            samples = tf.random.categorical(
                logits, num_samples=1, dtype=tf.int32)
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {logits.shape}')
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
                tf.concat([logis, logits], axis=1)
            ]

        def cond(*args):
            return True

        # _, _, tokens = tf.while_loop(
        _, _, tokens, logis = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
                tf.zeros((5, 5, 50257))
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(
                    hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        # return tokens
        return tokens, logis

def single_tokie (*, hparams, length, start_token=None,
                    batch_size=None, context=None):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

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
    return step(hparams, (context[:,:]), past=None)['logits']

    

def generate2(sess,
            run_name='run1',
            checkpoint_dir='checkpoint',
            model_name=None,
            model_dir='models',
            sample_dir='samples',
            return_as_list=False,
            truncate=None,
            destination_path=None,
            sample_delim='=' * 20 + '\n',
            prefix=None,
            seed=None,
            nsamples=1,
            batch_size=1,
            length=1023,
            temperature=0.7,
            top_k=0,
            top_p=0.0,
            include_prefix=True):
    """Generates text from a model loaded into memory.
    Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    """

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix == '':
        prefix = None

    if model_name:
        checkpoint_path = os.path.join(model_dir, model_name)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, run_name)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if prefix:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        context_tokens = enc.encode(prefix)

    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    logis = single_tokie(hparams=hparams, 
                            length=1023 - len(context_tokens),
                            start_token=None,
                            context=context,
                            batch_size=batch_size)
    #logis = tf.math.sigmoid(logis)


    f_me = sess.run(logis, feed_dict={
                    context: batch_size * [context_tokens]
                    })[0]


    print(f'f_me shaep: {f_me.shape}, context_tokens len: {len(context_tokens)}')
    enc = encoder.get_encoder(checkpoint_path)
    for offset in [-1,0,1]:
        for i, token in enumerate(context_tokens):
            try:
                print(token, enc.decode([token]), f_me[i+offset, token])
            except:
                pass

    plt.hist(f_me.flatten())
    plt.show()

    print(f_me.shape)
    print(f_me)
    return
                            


sess = start_tf_sess()
load_gpt2(sess)


output = generate2(sess, prefix='The cat is the horse of this country', include_prefix=False, return_as_list=True, length=100, batch_size=1, nsamples=5)
print(output)

