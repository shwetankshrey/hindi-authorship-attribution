import pickle
import nltk

tnt_pos = nltk.tag.tnt.TnT()
train_data = nltk.corpus.indian.tagged_sents('hindi.pos')
tnt_pos.train(train_data)

files = ["bhairav", "dharamveer", "premchand", "sharatchandra", "vibhooti"]
n = 1

piece_ngram_frequency = []
piece_author = []

pussydestroyerxx = 0

for file_name in files:
    pickle_file = open("../pickles/author_splits/" + file_name + ".pkl" , "rb")
    split_text = pickle.load(pickle_file)
    pickle_file.close()

    for text in split_text:
        tokens = nltk.word_tokenize(text)
        tagged = tnt_pos.tag(tokens)
        pussydestroyerxx += 1
        print(tagged)
        print(pussydestroyerxx)
        pos_tokens = list(tag[1] for tag in tagged)
        grams = nltk.ngrams(pos_tokens, n)
        piece_ngram_frequency.append(dict(nltk.FreqDist(grams)))
        piece_author.append(file_name)

corpus_ngram_frequency = {}

for ngf in piece_ngram_frequency:
    for ng in ngf.keys():
        if ng not in corpus_ngram_frequency:
            corpus_ngram_frequency[ng] = 0
        corpus_ngram_frequency[ng] += ngf[ng]

corpus_ngram_frequency = dict(sorted(corpus_ngram_frequency.items(), key=lambda x:x[1], reverse=True)[:500])

piece_frequencies = []

for ngf in piece_ngram_frequency:
    ng_freq_list = []
    for ng in corpus_ngram_frequency.keys():
        if ng in ngf:
            ng_freq_list.append(ngf[ng])
        else:
            ng_freq_list.append(0)
    piece_frequencies.append(ng_freq_list)

feature_vector = [piece_author, piece_frequencies]

pickle_file = open("../pickles/feature_vectors/pos_n_grams/" + str(n) + "grams.pkl" , "wb")
pickle.dump(feature_vector, pickle_file)
pickle_file.close()

