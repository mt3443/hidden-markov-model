import nltk
import string
import pickle

# get raw text from file
f = open('text/amazon_reviews.txt')
text = ' '.join(f.read().splitlines())
f.close()

# split text into sentences
sentences = nltk.sent_tokenize(text)

lines_to_remove = []
for i in range(len(sentences)):

	# remove all punctuation and set all words to lowercase
	sentences[i] = sentences[i].translate(str.maketrans('', '', string.punctuation)).lower()

	# record which lines are blank, so those can be removed
	if sentences[i] == '':
		lines_to_remove = [i] + lines_to_remove

# remove blank lines
for i in lines_to_remove:
	del sentences[i]

# preserve result
pickle.dump(sentences, open('pickle/sentences.p', 'wb'))