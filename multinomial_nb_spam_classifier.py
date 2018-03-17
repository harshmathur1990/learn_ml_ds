import json
import os
import math
import codecs
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from random import shuffle

samples_dir = 'samples/nb'
vocabulary = json.loads(open('dictionary/words_dictionary.json').read()).keys()


def train():
    classes = os.listdir(samples_dir)
    priors = {}
    total_samples = 0
    conditional_probability_of_term = defaultdict(dict)
    for _class in classes:
        print '############Training for {}#############'.format(_class)
        if _class == '.DS_Store':
            continue
        this_class_directory = samples_dir + '/' + _class
        sample_filenames = os.listdir(this_class_directory)
        shuffle(sample_filenames)
        sample_filenames = sample_filenames[:int(len(sample_filenames)*0.7)]
        total_samples += len(sample_filenames)
        priors[_class] = len(sample_filenames)

        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        text = u''
        for sample_filename in sample_filenames:
            if sample_filename == '.DS_Store':
                continue
            print '############Considering File {}#############'.format(sample_filename)
            try:
                f = codecs.open(this_class_directory + '/' + sample_filename, "r", "utf-8")
                try:
                    text += f.read()
                except Exception as exp:
                    print this_class_directory + '/' + sample_filename
            except Exception as e:
                f = codecs.open(this_class_directory + '/' + sample_filename, "r", "Windows-1252")
                try:
                    text += f.read()
                except Exception as exp:
                    print this_class_directory + '/' + sample_filename

        vectorizer.fit_transform(text.split(' '))

        total_score = 0.0

        total_keys = 0.0
        for word, score in vectorizer.vocabulary_.iteritems():

            conditional_probability_of_term[_class].update({
                word : score+1
            })

            total_score += score

            total_keys += 1

        for word, cp in conditional_probability_of_term[_class].iteritems():
            conditional_probability_of_term[_class][word] = float(cp)/float((total_score + total_keys))

    for _class, evidence_no in priors.iteritems():
        priors[_class] = float(evidence_no)/float(total_samples)

    return vocabulary, priors, conditional_probability_of_term


def apply_nb(sample, priors, conditional_probability_of_term):
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)

    text = u''
    print '#######Calculating for {}###########'.format(sample)
    try:
        f = codecs.open(sample, "r", "utf-8")
        try:
            text += f.read()
        except Exception as exp:
            print sample
    except Exception as e:
        f = codecs.open(sample, "r", "Windows-1252")
        try:
            text += f.read()
        except Exception as exp:
            print sample

    vectorizer.fit_transform(text.split(' '))

    max_class = None
    max_score = 0.0

    for _class, prior_pb in priors.iteritems():
        score = math.log(prior_pb)

        for word, sc in vectorizer.vocabulary_.iteritems():
            score += math.log(conditional_probability_of_term[_class][word])

        if not max_class or score > max_score:
            max_class = _class
            max_score = score

    return max_class, max_score


def test_accuracy():
    print '##############Training#################################'
    vocabulary, priors, conditional_probability_of_term = train()

    print '##############Training Done############################'
    classes = os.listdir(samples_dir)

    result = defaultdict(list)

    for _class in classes:
        if _class == '.DS_Store':
            continue
        this_class_directory = samples_dir + '/' + _class
        sample_filenames = os.listdir(this_class_directory)
        shuffle(sample_filenames)
        sample_filenames = sample_filenames[-int((len(sample_filenames) * 0.3)):]

        for filename in sample_filenames:
            if filename == '.DS_Store':
                continue

            predicted_class, score = apply_nb(this_class_directory + '/' + filename, priors, conditional_probability_of_term)

            _result = {
                'class': _class,
                'filename': filename,
                'predicted_class': predicted_class,
                'score': score
            }

            if predicted_class == _class:
                result['success'].append(_result)
            else:
                result['failure'].append(_result)

    total_tests = len(result['success']) + len(result['failure'])

    return total_tests, float(len(result['success']))/float(total_tests)


if __name__ == '__main__':
    total_tests, accuracy = test_accuracy()
    print accuracy
