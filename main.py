#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gengram
from classifier import ClassifierClass
import re


def tokenize_sentence(sentence):
	s = re.sub('[()]', r'', generated)  # remove certain punctuation chars
	s = re.sub('([.-])+', r'\1', s) # collapse multiples of certain chars
	s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
	return s.split()

def tokenize_data(filenames):
	data = []
	for filename in filenames:
		rawdata = list(open(filename, 'r', encoding="utf8"))
		for s in rawdata:
			if(len(s) > 1): #remove whitelines
				s = re.sub('[()]', r'', s)  # remove certain punctuation chars
				s = re.sub('([.-])+', r'\1', s) # collapse multiples of certain chars
				s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
				s = s.split()
				data.append({"class":filename, "words":s})
	return data;

if __name__ == "__main__":
	classifierdata = tokenize_data(["clickbait_data", "non_clickbait_data"])

	clsfier = ClassifierClass.train(classifierdata)
	corpus = gengram.preprocess_corpus("clickbait_data")
	generated = gengram.gengram_sentence(corpus)
	print(generated)
	probability = clsfier.predict(tokenize_sentence(generated))
	print(probability)
	print(max(probability, key=probability.get))


	
	print("-"*30)
	best = ""
	lowest = -10000000
	for i in range(0, 10):
		generated = gengram.gengram_sentence(corpus)
		probability = clsfier.predict(tokenize_sentence(generated))
		if(lowest < probability["clickbait_data"]):
			print(generated)
			best = generated
			lowest = probability["clickbait_data"]
	print("-"*30)
	print(best, lowest)
