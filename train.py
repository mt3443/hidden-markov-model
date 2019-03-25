import pickle
from hmm import HMM

# get clean, preformatted text from corpus
print('Loading sample text... ', end='')
text = pickle.load(open('pickle/sentences.p', 'rb'))
print('done')

model = HMM(n_hidden_states=8)
model.train(text)