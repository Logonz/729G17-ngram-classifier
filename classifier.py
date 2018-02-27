#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

class ClassifierClass(object):

    def predict(self, d):
        result = {}
        for c in self.classes:
            result[c] = 0
            
        for word in d:
            for c in self.classes:
                try: #test voc
                    pw = self.pw[c][word]
                    result[c] += pw
                except Exception as ex:
                    g = 0
 
        for key, value in result.items():
            result[key] = value+self.pc[key]
       	#returnkey = max(result, key=result.get)
        return result

    def getbaseline(self, data):
        classes = {}

        for d in data:
            if(d["class"] in classes):
                classes[d["class"]] += 1
            else:
                classes[d["class"]] = 1
        self.mostfreq = max(classes, key=classes.get)
        return classes

    @classmethod
    def train(cls, rawdata, k=1):
        # The following line creates a new object of type Classifier:
        classifier = cls()
        # The next few lines initialise the four attributes of the classifier:
        classes = {}
        classifier.classes = set()
        classifier.vocabulary = set()
        classifier.pc = {} 
        classifier.pw = {}
        amount_of_words = 0
        #Populate vocabulary and classes
        for d in rawdata:
            #Makes the classes and how many there are of each
            if(d["class"] in classes):
                classes[d["class"]] += 1
            else:
                classes[d["class"]] = 1
                
            #Popualted the vocabulary
            for word in d["words"]:
                amount_of_words += 1
                classifier.vocabulary.add(word)
                
                #Populate PW
                #If not in list, add it and set the value to 1 and populate the with the word
                if(d["class"] not in classifier.pw):
                    classifier.pw[d["class"]] = {word:1}
                else:
                    #The word exists, add 1
                    if(word in classifier.pw[d["class"]]):
                        classifier.pw[d["class"]][word] += 1
                    else:
                        #The word didn't exist in the dict in the set so add it and set it to 1
                        classifier.pw[d["class"]][word] = 1
        #Set the classes and pc
        for typeclass, count in classes.items():
            classifier.classes.add(typeclass)
            classifier.pc[typeclass] = math.log(count/len(rawdata))
            
        #Smooth
        for word in classifier.vocabulary:
            for typeclass, count in classes.items():
                if(word not in classifier.pw[typeclass]):
                    classifier.pw[typeclass][word] = k
                else:
                    classifier.pw[typeclass][word] += k

        #Estimate word probablities
        for typeclass, typedata in classifier.pw.items():
            print(typeclass, len(typedata), sum(typedata.values()))
            length = sum(typedata.values())
            for word, count in typedata.items():
                typedata[word] = math.log(count/length) #The probability, not in Log format!!
                
        print("Used k:", k)
        return classifier

def accuracy(classifier, data):
    tot = len(data)
    correct = 0
    for d in data:
        #print(classifier.predict(d["words"]), d["class"])
        pred = classifier.predict(d["words"])
        predclass = max(pred, key=pred.get)
        if(predclass == d["class"]):
            correct += 1
    
    return correct/tot

def precision(classifier, c, documents):
    tot = len(documents)
    truepos = 0
    falsepos = 0
    for d in documents:
        predicted = classifier.predict(d["words"])
        predictedclass = max(predicted, key=predicted.get)
        if(d["class"] == predictedclass and d["class"] == c):
            truepos += 1
        elif(d["class"] != predictedclass and d["class"] != c):
            falsepos += 1
            
        
        if(classifier.predict(d["words"]) == None):
            return(float('NaN'))
    
    return(truepos/(truepos+falsepos))

def recall(classifier, c, documents):
    tot = len(documents)
    truepos = 0
    falseneg = 0
    for d in documents:
        predicted = classifier.predict(d["words"])
        predictedclass = max(predicted, key=predicted.get)
        if(d["class"] == predictedclass and d["class"] == c):
            truepos += 1
        elif(d["class"] != predictedclass and predictedclass != c):
            falseneg += 1
            
        
        if(classifier.predict(d["words"]) == None):
            return(float('NaN'))

    return(truepos/(truepos+falseneg))

def F1(p, r):
    return (2*p*r)/(p+r)