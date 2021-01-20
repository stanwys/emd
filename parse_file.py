import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer

PHRASES = ["good","best","love","suck","great","fun","enjoy","favorite","entertain"]#,"averag"]#,"mediocre","easy"]

STARS = ['1','2','4']#'3','4','5']
STARS_MAP = {
    '1' : 'one', '2' : 'two', '3' : 'three', '4': 'four', '5':'five'
}

def countOccurencesInString(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    counter = 0
    i = s.find(p)
    while i != -1:
        counter += 1
        i = s.find(p, i+1)
    return counter

def calculateStarAttribute(num_stars, allText):
    string1 = num_stars+" star"
    string2 = STARS_MAP[num_stars] + " star"
    if string1 in allText or string2 in allText:
        return 1
    else:
        return 0

def parseHelpful(helpful_field):
    nums = helpful_field[1:-1].split(",")
    return nums

def parseRecord(dfrecord):
    record = []
    help = parseHelpful(dfrecord["helpful"])
    for ob in help:
        record.append(ob)
    record.append(dfrecord["unixReviewTime"])

    return record

def analyseFile(filedir):
    df = pd.read_csv(filedir, sep=',', error_bad_lines=False)
    values = df.values
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    dictionary = {}
    word_count = {}
    score_words = {}
    score_keys = [1,2,3,4,5]
    phrases_keys = ["good","best","love","suck","great","fun","enjoy",
                    "favorite","entertain","averag","mediocre","easy","hate","don't like"]
    phrases_stars = ["1star","2star","3star","4star","5star"]

    for pkey in phrases_keys:
        dictionary[pkey] = {}
        word_count[pkey] = {}
        for skey in score_keys:
            dictionary[pkey][skey] = 0
            word_count[pkey][skey] = 0

    for star_key in phrases_stars:
        word_count[star_key] = {}
        for skey in score_keys:
            word_count[star_key][skey] = 0

    for skey in score_keys:
        score_words[skey]={}

    for row in values:
        all_text = ""
        if (type(row[4]) is str):
            all_text = row[4]

        if (type(row[5]) is str):
            all_text += (" "+row[5])

        skey = int(row[8])
        all_text = all_text.lower()

        for star_key in phrases_stars:
            word_count[star_key][skey] += calculateStarAttribute(star_key[0], all_text)

        tokens = tokenizer.tokenize(all_text.lower())
        for token in tokens:
            if token in score_words[skey].keys():
                score_words[skey][token] += 1
            else:
                score_words[skey][token] = 1

        for pkey in phrases_keys:
            word_count[pkey][skey] += countOccurencesInString(pkey, all_text)
            if pkey in all_text:
                dictionary[pkey][skey] += 1

    #listofTuples = sorted(word_count.items(), reverse=True, key=lambda x: x[1])
    for key in score_words.keys():
        listofTuples = sorted(score_words[key].items(), reverse=True, key=lambda x: x[1])
        print(listofTuples[0])

    print(dictionary)

    df = pd.DataFrame(data=word_count, index=score_keys).T
    df.to_csv('multi_value_vector_word_count.csv') #mode='a', header=False)

def parseRecord2(row, binary_value):
    record = []

    help = parseHelpful(row[3])
    for ob in help:
        record.append(int(ob))

    all_text = ""
    if (type(row[4]) is str):
        all_text = row[4]

    if (type(row[5]) is str):
        all_text += (" "+row[5])

    all_text = all_text.lower()

    for pkey in PHRASES:
        if binary_value == True:
            if pkey in all_text:
                record.append(1)
            else:
                record.append(0)
        else:
            record.append(countOccurencesInString(pkey, all_text))


    for star in STARS:
        record.append(calculateStarAttribute(star, all_text))

    return record

def parseCSV(filedir, binary_value = False):
    df = pd.read_csv(filedir, sep=',', error_bad_lines=False)

    values = df.values
    Y = np.asarray(values[:,-1],dtype=int)
    records = [parseRecord2(row, binary_value) for row in values]

    X = np.asarray(records, dtype=np.int)

    return X,Y