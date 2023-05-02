import random
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization



class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
        })
        return config


embed_dim = 256
latent_dim = 2048
num_heads = 8


vocab_size = 15000
sequence_length = 20
batch_size = 64

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)




#LOAD SAVED MODEL

import re
def load_model_files(path):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")

    try:

        @tf.keras.utils.register_keras_serializable()
        def custom_standardization(input_string):
            lowercase = tf.strings.lower(input_string)
            return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
    except: pass

    inp_vectorization_ = tf.keras.models.load_model(path+"/"+"inp_vectorization_layer").layers[0]
    out_vectorization_ = tf.keras.models.load_model(path+"/"+"out_vectorization_layer").layers[0]

    #LOAD SAVED MODEL

    from tensorflow import keras
    transformer = tf.keras.models.load_model(path+"/"+'bliss_to_english-10-0.89.hdf5', custom_objects={'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder':TransformerEncoder, 'TransformerDecoder':TransformerDecoder})

    return inp_vectorization_, out_vectorization_, transformer





def decode_sequence(input_sentence, inp_vectorization_,out_vectorization_, transformer):

    out_vocab = out_vectorization_.get_vocabulary()
    out_index_lookup = dict(zip(range(len(out_vocab)), out_vocab))
    max_decoded_sentence_length = 20

    tokenized_input_sentence = inp_vectorization_([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = out_vectorization_([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = out_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence




## change_tense

import string

from pattern.en import conjugate, PAST, PRESENT, SINGULAR, PLURAL
import spacy
from spacy.symbols import NOUN



SUBJ_DEPS = {'agent', 'csubj', 'csubjpass', 'expl', 'nsubj', 'nsubjpass'}

nlp = spacy.load('en_core_web_sm')


def _get_conjuncts(tok):
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights
            if right.dep_ == 'conj']


def is_plural_noun(token):
    """
    Returns True if token is a plural noun, False otherwise.
    Args:
        token (``spacy.Token``): parent document must have POS information
    Returns:
        bool
    """
    if token.doc.has_annotation("TAG") is False:
        raise ValueError('token is not POS-tagged')
    return True if token.pos == NOUN and token.lemma != token.lower else False


def get_subjects_of_verb(verb):
    if verb.dep_ == "aux" and list(verb.ancestors):
        return get_subjects_of_verb(list(verb.ancestors)[0])
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts
             if tok.dep_ in SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    if not len(subjs):
        ancestors = list(verb.ancestors)
        if len(ancestors) > 0:
            return get_subjects_of_verb(ancestors[0])
    return subjs


def is_plural_verb(token):
    if token.doc.has_annotation("TAG") is False:
        raise ValueError('token is not POS-tagged')
    subjects = get_subjects_of_verb(token)
    if not len(subjects):
        return False
    plural_score = sum([is_plural_noun(x) for x in subjects])/len(subjects)

    return plural_score > .5

def preserve_caps(word, newWord):
    """Returns newWord, capitalizing it if word is capitalized."""
    if word[0] >= 'A' and word[0] <= 'Z':
        newWord = newWord.capitalize()
    return newWord

def change_tense(text, to_tense, nlp=nlp):
    """Change the tense of text.
    Args:
        text (str): text to change.
        to_tense (str): 'present','past', or 'future'
        npl (SpaCy model, optional):
    Returns:
        str: changed text.
    """
    tense_lookup = {'future': 'inf', 'present': PRESENT, 'past': PAST}
    tense = tense_lookup[to_tense]

    doc = nlp(text)

    out = list()
    out.append(doc[0].text)
    words = []
    for word in doc:
        words.append(word)
        if len(words) == 1:
            continue
        if (words[-2].text == 'will' and words[-2].tag_ == 'MD' and words[-1].tag_ == 'VB') or \
                        words[-1].tag_ in ('VBD', 'VBP', 'VBZ', 'VBN') or \
                (not words[-2].text in ('to', 'not') and words[-1].tag_ == 'VB'):

            if words[-2].text in ('were', 'am', 'is', 'are', 'was') or\
                    (words[-2].text == 'be' and len(words) > 2 and words[-3].text == 'will'):
                this_tense = tense_lookup['past']
            else:
                this_tense = tense

            subjects = [x.text for x in get_subjects_of_verb(words[-1])]
            if ('I' in subjects) or ('we' in subjects) or ('We' in subjects):
                person = 1
            elif ('you' in subjects) or ('You' in subjects):
                person = 2
            else:
                person = 3
            if is_plural_verb(words[-1]):
                number = PLURAL
            else:
                number = SINGULAR
            if (words[-2].text == 'will' and words[-2].tag_ == 'MD') or words[-2].text == 'had':
                out.pop(-1)
            if to_tense == 'future':
                if not (out[-1] == 'will' or out[-1] == 'be'):
                    out.append('will')
                # handle will as a noun in future tense
                if words[-2].text == 'will' and words[-2].tag_ == 'NN':
                    out.append('will')
            #if word_pair[0].dep_ == 'auxpass':
            oldWord = words[-1].text
            out.append(preserve_caps(oldWord, conjugate(oldWord, tense=this_tense, person=person, number=number)))
        else:
            out.append(words[-1].text)

        # negation
        if words[-2].text + words[-1].text in ('didnot', 'donot', 'willnot', "didn't", "don't", "won't"):
            if tense == PAST:
                out[-2] = 'did'
            elif tense == PRESENT:
                out[-2] = 'do'
            else:
                out.pop(-2)

        # future perfect, and progressives, but ignore for "I will have cookies"
        if words[-1].text in ('have', 'has') and len(list(words[-1].ancestors)) and words[-1].dep_ == 'aux':
            out.pop(-1)

    text_out = ' '.join(out)

    # Remove spaces before/after punctuation:
    for char in string.punctuation:
        if char in """(<['""":
            text_out = text_out.replace(char+' ', char)
        else:
            text_out = text_out.replace(' '+char, char)

    for char in ["-", "“", "‘"]:
        text_out = text_out.replace(char+' ', char)
    for char in ["…", "”", "'s", "n't"]:
        text_out = text_out.replace(' '+char, char)

    return text_out
