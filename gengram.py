#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random, re

SEP = " " # token separator symbol

def make_ngrams(tokens, N):
    """ Returns a list of N-long ngrams from a list of tokens """

    ngrams = []
    for i in range(len(tokens)-N+1):
        ngrams.append(tokens[i:i+N])
    return ngrams


def ngram_freqs(ngrams):
    """ Builds dict of TOKEN_SEQUENCEs and NEXT_TOKEN frequencies """

    ### has form TOKEN_SEQUENCE : DICT OF { NEXT_TOKEN : COUNT }
    ###      e.g        "a b c" : {"d" : 4, "e" : 2, "f" : 6 }
    counts = {}

    # Using example of ngram "a b c e" ...
    for ngram in ngrams:
        token_seq  = SEP.join(ngram[:-1])   # "a b c"
        last_token = ngram[-1]              # "e"

        # create empty {NEXT_TOKEN : COUNT} dict if token_seq not seen before
        if token_seq not in counts:
            counts[token_seq] = {};

        # initialize count for newly seen next_tokens
        if last_token not in counts[token_seq]:
            counts[token_seq][last_token] = 0;

        counts[token_seq][last_token] += 1;

    return counts;


def next_word(text, N, counts):
    """ Outputs the next word to add by using most recent tokens """

    token_seq = SEP.join(text.split()[-(N-1):]);
    choices = counts[token_seq].items();

    # make a weighted choice for the next_token
    # [see http://stackoverflow.com/a/3679747/2023516]
    total = sum(weight for choice, weight in choices)
    r = random.uniform(0, total)
    upto = 0
    for choice, weight in choices:
        upto += weight;
        if upto > r: return choice
    assert False                            # should not reach here


def preprocess_corpus(filename):
    s = open(filename, 'r', encoding="utf8").read()
    s = re.sub('[()]', r'', s)                              # remove certain punctuation chars
    s = re.sub('([.-])+', r'\1', s)                         # collapse multiples of certain chars
    s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
    s = ' '.join(s.split()).lower()                         # remove extra whitespace (incl. newlines)
    return s;


def postprocess_output(s):
    s = re.sub('\\s+([.,!?])\\s*', r'\1 ', s)                       # correct whitespace padding around punctuation
    s = s.capitalize();                                             # capitalize first letter
    s = re.sub('([.!?]\\s+[a-z])', lambda c: c.group(1).upper(), s) # capitalize letters following terminated sentences
    return s

def sentence_wordcount(sentence):
	s = re.sub('[()]', r'', sentence)  # remove certain punctuation chars
	s = re.sub(r"(\.)|(\?)|(\!)|(\,)|(\")|(\')", r"", s) # remove punctuations because we use this as a wordcount
	s = re.sub('([.-])+', r'\1', s) # collapse multiples of certain chars
	s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
	return s.split()

#This generates a good start sequence
def gengram_startseq(data):
	start_seq = ".!?"
	while True:
		correct = True
		start_seq = random.choice(list(data.keys()));
		for char in start_seq:
			if(len(re.findall(r"[a-zA-z]|[1-9]|\ |(\")|(\')", char)) == 0):# Do not have any other characters in the first words.
				correct = False
				break
		if(correct):
			return start_seq
	return start_seq #Fallback, just return something!

def gengram_sentence(corpus, N=3, sentence_count=1, start_seq=None, min_words=4):
    """ Generate a random sentence based on input text corpus """

    ngrams = make_ngrams(corpus.split(SEP), N)
    counts = ngram_freqs(ngrams)

    if start_seq is None:
    	start_seq = gengram_startseq(counts) #More advanced start_seq generator.
    rand_text = start_seq.lower();

    sentences = 0;
    while sentences < sentence_count:
        rand_text += SEP + next_word(rand_text, N, counts);
        if(len(sentence_wordcount(rand_text)) >= min_words): #Is it less than words? Dont finish sentence
       		sentences += 1 if rand_text.endswith(('.','!', '?')) else 0

    return postprocess_output(rand_text);


if __name__ == "__main__":

    corpus = preprocess_corpus("clickbait_data")
    print(gengram_sentence(corpus))