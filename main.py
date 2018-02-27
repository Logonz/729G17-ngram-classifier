#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gengram
import classifier# import ClassifierClass,
import re
import operator


def tokenize_sentence(sentence):
	s = re.sub('[()]', r'', sentence)  # remove certain punctuation chars
	s = re.sub('([.-])+', r'\1', s) # collapse multiples of certain chars
	s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
	return s.split()

def tokenize_data(filenames):
	data = {}

	for filename in filenames:
		subdata = []
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
				subdata.append({"class":filename, "words":s})
		print("Avg sentence length:", filename, total/count)
		data[filename] = subdata
	return data;



def split_data(data, split_amount=0.8):
	retdata = {}
	for filename, subdata in data.items():
		retdata[filename] = {}
		retdata[filename]["train"] = subdata[:int(len(subdata)*split_amount)]
		retdata[filename]["test"] = subdata[int(len(subdata)*(1-split_amount)):]
		#print(len(retdata[filename]["test"]))

	return retdata

if __name__ == "__main__":
	classifierdata = tokenize_data(["clickbait_data", "non_clickbait_data"])
	data = split_data(classifierdata)

	clsfier = classifier.ClassifierClass.train(data["clickbait_data"]["train"]+data["non_clickbait_data"]["train"], k=0.01)
	combined = []
	print("-"*30)
	for filename in data.keys():
		combined += data[filename]["test"]
	print("%s accuracy" % (classifier.accuracy(clsfier, combined)*100))
	for filename in data.keys():
		pre = classifier.precision(clsfier, filename, data[filename]["test"])
		recall = classifier.recall(clsfier, filename, data[filename]["test"])
		f1 = classifier.F1(pre, recall)

		print("\nClass:%s - recall:%s, precision:%s, F1:%s" % (filename, (recall*100), (pre*100), (f1*100)))
	print("\nbaseline", clsfier.getbaseline(data["clickbait_data"]["train"]+data["non_clickbait_data"]["train"]))
	print("-"*30)


	corpus = gengram.preprocess_corpus("clickbait_data")
	
	#junk code
	#generated = gengram.gengram_sentence(corpus)
	#print(generated)
	#probability = clsfier.predict(tokenize_sentence(generated))
	#print(probability)
	#print(max(probability, key=probability.get))
	#print("test")
	
	#print("-"*30)
	sentences = {}
	skipped = 0
	for i in range(0, 100):
		generated = gengram.gengram_sentence(corpus)
		probability = clsfier.predict(tokenize_sentence(generated))
		if(i%5==0):
			print("i:%s \t %s" % (i, generated))
		cbnormalized = probability["clickbait_data"]/len(gengram.sentence_wordcount(generated))
		#ncbnormalized = probability["non_clickbait_data"]/len(gengram.sentence_wordcount(generated))
		#if(cbnormalized > ncbnormalized):
		sentences[generated] = cbnormalized
		#else:
		#	print("Threw away", "i: %s cb:%s : ncb:%s" % (i, str(cbnormalized)[:5], str(ncbnormalized)[:5]), generated)
		#	skipped += 1

	print("-"*30)
	sorted_x = sorted(sentences.items(), key=operator.itemgetter(1), reverse=True)
	for sentence in sorted_x[:10]:
		print(sentence)

	#print("Skipped: "+ str(skipped))
	#print(sorted_x)
	#print("\n".join(sorted_x[:5]))

