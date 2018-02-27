#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gengram
from classifier import ClassifierClass
import re


def tokenize_sentence(sentence):
	s = re.sub('[()]', r'', sentence)  # remove certain punctuation chars
	s = re.sub('([.-])+', r'\1', s) # collapse multiples of certain chars
	s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
	return s.split()

def tokenize_data(filenames):
	data = []

	for filename in filenames:
		rawdata = list(open(filename, 'r', encoding="utf8"))
		total = 0
		count = 0
		for s in rawdata:
			if(len(s) > 1): #remove whitelines
				count += 1
				s = re.sub('[()]', r'', s)  # remove certain punctuation chars
				s = re.sub('([.-])+', r'\1', s) # collapse multiples of certain chars
				s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
				s = s.split()
				total += len(s) #check the sentence word count
				data.append({"class":filename, "words":s})
		print("Avg sentence length:", filename, total/count)
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
	#print("test")
	
	print("-"*30)
	best = ""
	lowest = -10000000
	for i in range(0, 100):
		generated = gengram.gengram_sentence(corpus)
		probability = clsfier.predict(tokenize_sentence(generated))
		print(generated)
		if(lowest < probability["clickbait_data"]):
			best = generated
			lowest = probability["clickbait_data"]
	print("-"*30)
	print(best, lowest)
