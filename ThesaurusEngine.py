#!/usr/bin/env python
# coding: utf-8




# ### Preamble

import re
import nltk
import logging

from numpy.random import choice
from nltk.corpus import wordnet as wn
from lemminflect import getInflection,getLemma

Tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
logger = logging.getLogger()


# apropos punctuation
PunctStr = '!#$%&\'*+-/=/?@\\^_`|~'
PunctSet = set(PunctStr)

DelimStr = ',.;:'
DelimSet = set(DelimStr)

ParenStr = '«»()[]{}<>"'
ParenSet = set(ParenStr)


# words with these WordNet tags can be synonymised:
# n: noun
# v: verb
# a: adjective
# r: adverb
ValidWNTags = ["n", "v", "a", "r"]


# vowels for reference:
Vowels = ("a", "e", "i", "o", "u")


# NLTK labels these immutables wrong, so I must manually clean up.
TheCracks = ["have", "has", "had", "be", "is", "was", "are", "been", "do", "does", "did"]
CracksDict = {'have': 'have', 'has': 'have', 'had': 'have', 'be': 'be', 'is': 'be', 'was': 'be', 'are': 'be', 'been': 'be', 'do': 'do', 'does': 'do', 'did': 'do'}


# fodder:
text0 = """'One day, the people (that don't even like / love you) are going to tell everyone how they met you' -- Mr. Johnny Depp. What a terrible quote from a hot.com person."""

text1 = """word replacement engine"""


# ### Functions

# supplementary
def _translateTag(tag , fmt):
    """ Convert tags from universal tagset to either WordNet or LemmInflect format. """
    # shouldn't handle MOD or AUX tags, those should bypass this as they are immutable.
    otag = str()
    if tag.startswith("V"):
        otag = "VERB" if fmt=="LI" else wn.VERB
    elif tag.startswith("J"):
        otag = "ADJ"  if fmt=="LI" else wn.ADJ
    elif tag.startswith("R"):
        otag = "ADV"  if fmt=="LI" else wn.ADV
    elif tag.startswith("N"):
        otag = "NOUN" if fmt=="LI" else wn.NOUN
    return otag


def _synoname(wordobj):
    """ Return just the synonym (no metadata) from a WordNet synset object. """
    # surely there's an 'official' way to retrieve the word?
    return wordobj.name().split('.')[0] if type(wordobj) is nltk.corpus.reader.wordnet.Synset else wordobj




# principal
def extract(raw):
    """ Extract information from text at the word level, unaggregated. """
    # builds [("word1","tag1") , ("word2","tag2") , ... ]
    wordtags = nltk.pos_tag( nltk.word_tokenize(raw) )
    return list(map( lambda wt:(wt[0].lower() , wt[1]) , wordtags ))


def lemmatise(wordtags):
    """ Find and filter valid lemmas. """
    for word,tag in wordtags:
        liTag = _translateTag(tag,"LI")
        if (tag in ["NNP"]) or (word in TheCracks) or (len(word) < 3) or not liTag:
            lemma = word
        else:
            lemma = getLemma(word , upos=liTag)[0]
            
        yield lemma,tag


def synonymise(wordtags):
    """ Find and filter valid synonyms"""
    for word,tag in wordtags:
        # again, the following get a free pass because NLP is still neotenous:
        if (tag in ["NNP"]) or (word in TheCracks) or (len(word) < 3):
            yield word,tag
            continue
            
        # facilitate discussion between libraries:
        wnTag = _translateTag(tag,"WN")
        
        # collect possible synonyms:
        synonyms = {lemma for synset in wn.synsets(word,pos=wnTag) for lemma in synset.lemma_names()}
        
        # POS tags must be mutable (i.e. exclude proper nouns, pronouns, etc.):
        synonyms = list(filter( lambda syn:(wnTag in ValidWNTags) , synonyms ))
        
        # collocations will have an underscore rather than a space, let's fix that:
        synonyms = list(map( lambda syn:_synoname(syn).replace("_"," ") , synonyms ))
        
        #custom distribution so earlier synonyms are more favourable:
        # [1]
        # [1/2  1/2]
        # [1/2  1/4  1/4]
        # [1/2  1/4  1/8  1/8]
        # [1/2  1/4  1/8  1/16  1/16]a\
        dist = [2**(-i-1) for i in range(len(synonyms)-1)]
        dist.append( 1 - sum(dist) )
        
        synonym = choice(synonyms , p=dist) if synonyms else word
        
        yield synonym.lower(),tag


def inflectivise(wordtags):
    """ Inflect list of (word,tag) tuples correctly. """
    wordtaglist = list(wordtags)
    for i,(word,tag) in enumerate(wordtaglist):
        # conjugation and declension:
        if (tag in ["NNP"]) or (word in TheCracks) or (len(word) < 3):
            yield word,tag
            continue
        
        if word in ["a","an"]:
            inflected = "a"
            if wordtaglist[i+1][0].startswith(Vowels):
                inflected = "an"
        elif " " not in word:
            inflected = getInflection(word,tag=tag)
            inflected = word if not inflected else inflected[-1]
        else:
            # only conjugate the relevant part of a collocation:
            # e.g. "run across"  ->  "ran across"  NOT  "run acrossed"
            for subword,subtag in nltk.pos_tag( nltk.word_tokenize(word) ):
                if _translateTag(subtag,"WN") in ValidWNTags:
                    infl = getInflection(subword,tag=tag)[-1]
                    inflected = word.replace(subword , infl)
        
        yield inflected,tag


def assemble(wordtags):
    """ Autobots, assemble! """
    first = True
    logging.disable(logging.CRITICAL)
    for word,tag in wordtags:
        # truecasing proper nouns and "I":
        word = word.title() if (tag in ["NNP","NNPS"]) or (word == "i") else word
        
        # conditions in case of contractions or punctuation:
        wordset = set(word)
        hasPunct = bool( wordset & PunctSet )
        
        # without apostrophe, contractions have no punctuation:
        isContraction = hasPunct and not bool( (wordset - {"'"}) & PunctSet )
        isClauseDelim = not bool( wordset - DelimSet )
        isParentheses = not bool( wordset - ParenSet )
        
        if first:
            first = False
            draft = word
        elif isContraction or isClauseDelim:
            draft += word
        else:
            draft += " " + word
    
    # truecasing sentences:
    sentences = list()
    for sentence in Tokenizer.tokenize(draft):
        words = nltk.word_tokenize(sentence)
        sentences.append(sentence.replace( words[0] , words[0].title() , 1 ))
    final = ' '.join(sentences)
    
    # spacing surrounding parenthesese:
    final = re.sub( r'''(?<=[{[(])\s+     # ((any single opening bracket))  preceding  (at least one whitespace)
                        |                 # or
                        \s+(?=[]})])      # (at least one whitespace)  preceding  ((any single closing bracket))
                        ''', '' , final , flags=re.VERBOSE)

    return final



# ### Main

if __name__ == '__main__':
    prompt = "Enter grammatically correct text below so I can synonymise it, or enter \"file\" (no quotes) to read from input.txt:\n"
    which = input(prompt)
    if which.lower() == "file":
        with open('input.txt','r') as file:
            text = file.read()
    else:
        text = which
    print("\nORIGINAL:\n" + text + "\n")

    parsedText = extract(text)

    lemmaText = lemmatise(parsedText)

    synonymText = synonymise(lemmaText)

    inflectedText = inflectivise(synonymText)

    print("\nFINAL:\n" + assemble(inflectedText))
    input("\n\n\nPress enter to exit.")
