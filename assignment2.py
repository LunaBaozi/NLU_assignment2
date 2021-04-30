import spacy
import conll
import numpy as np
from spacy.training import Alignment
from spacy.tokens import Doc
from sklearn.metrics import classification_report
from pprint import pprint

# Loading the language model
nlp = spacy.load('en_core_web_sm')

## Reading training data
train = conll.read_corpus_conll('./conll2003/train.txt', ' ')
## Reading test data
train = conll.read_corpus_conll('./conll2003/test.txt', ' ')


## Dictionary for labels conversion
## Keys: spacy labels
## Values: coNLL labels
labels = {'PERSON': 'PER',
    'NORP': 'MISC',
    'FAC': 'LOC',
    'ORG': 'ORG',
    'GPE': 'LOC',
    'LOC': 'LOC',
    'PRODUCT': 'MISC',
    'EVENT': 'MISC',
    'WORK_OF_ART': 'MISC',
    'LAW': 'MISC',
    'LANGUAGE': 'MISC',
    'DATE': 'MISC',
    'TIME': 'MISC',
    'PERCENT': 'MISC',
    'MONEY': 'MISC',
    'QUANTITY': 'MISC',
    'ORDINAL': 'MISC',
    'CARDINAL': 'MISC',
    'O': 'O',
    '': ''}


def extract_tokens(corpus) -> list:

    # The function reconstructs the sentences present in the
    # coNLL2003 file, without any manipulation
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists containing the reconstructed sentences

    return([[''.join(sent[i][0]) for i in range(len(sent))] for sent in corpus])  


def extract_tokens_clean(corpus) -> list:

    # The function reconstructs the sentences present in the
    # coNLL2003 file, deleting what is not a sentence
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists containing the reconstructed sentences

    ex_tokens = extract_tokens(corpus)
    return([ex_tokens[i] for i in range(len(ex_tokens)) if ex_tokens[i][0] != '-DOCSTART-'])


def extract_tags(corpus) -> list:

    # The function extracts the tokens and relative IOB tags
    # from the input corpus file, without any manipulation
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists, containing tuples arranged like this: ('token', 'iob')

    return([[(text, iob) for text, pos, ch, iob in sent] for sent in corpus])


def extract_tags_clean(corpus) -> list:
    
    # The function extracts the tokens and relative IOB tags
    # from the input corpus file, deleting what is not a sentence
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists, containing tuples arranged like this: ('token', 'iob')

    words_plus_tags = extract_tags(corpus)
    return([words_plus_tags[i] for i in range(len(words_plus_tags)) if words_plus_tags[i][0][0] != '-DOCSTART-'])


def clean_sents(corpus) -> list:

    # The function reconstructs the sentences as strings and with 
    # the correct distribution of punctuation
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists containing string sentences

    # Initialize empty list
    cleansents = []

    # Retrieve tokens and IOB tags as outputted by the called function
    base = extract_tags_clean(corpus)

    # Iterate over sentences
    for sent in base:

        # Initialize empty list to store sentences
        txt = ''
        txt += sent[0][0]

        # Iterate over the tokens of each sentences
        for j in range(1, len(sent)):

            # If the element is in the set of punctuation...
            if sent[j][0] in ".,:;'":

                # ...paste it directly to the previous string
                txt += sent[j][0] 
            else:
                txt += ' ' + sent[j][0]

        # Append the reconstructed sentence as a string        
        cleansents.append([txt])  

    return cleansents


def spacy_on_cleansents_text(corpus) -> list:

    # The function applies spacy's tokenization to the
    # reconstructed sentences 
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists containing tokens.text

    base = clean_sents(corpus)
    return([[token.text for token in nlp(sent[0])] for sent in base])


def spacy_on_cleansents_token(corpus) -> list:

    # The function applies spacy's tokenization to the
    # reconstructed sentences
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return:a list of lists containing tokens

    base = clean_sents(corpus)
    return([[token for token in nlp(sent[0])] for sent in base])


def get_whitespaces(corpus) -> list:

    # The function retrieves the whitespaces of the corpus 
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists containing tokens whitespaces

    spacy_tokens = spacy_on_cleansents_token(corpus)
    return([[bool(token.whitespace_) for token in sent] for sent in spacy_tokens])


def get_spacy_tags(corpus) -> list:

    # The function reconstructs the entities of the tokens
    # constituting the corpus, without further manipulation
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists containing tuples with relevant objects

    cleansents = clean_sents(corpus)
    return([[(token.text, token.ent_iob_+'-'+labels[token.ent_type_]) for token in nlp(sent[0])] for sent in cleansents])


def get_spacy_tags_clean(corpus) -> list:

    # The function reconstructs the entities of the tokens
    # constituting the corpus, with further manipulation
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list of lists containing tuples with relevant objects

    spacy_plus_tags = get_spacy_tags(corpus)
    return([[(tup[0], 'O') if tup[1] == 'O-' else (tup[0], tup[1]) for tup in sent] for sent in spacy_plus_tags])


def get_spacy_alignment(ref, hyp) -> list:

    # The function returns the aligned tokenization between
    # a reference corpus and a hypothesis corpus
    # :param corpus: the corpus in coNLL format of which we want to reconstruct sentences
    # :return: a list cotaining alignment objects

    alignment = list()

    # Iterate over pairs of objects
    for i, j in zip(ref, hyp):

        # Compute the alignment between the two objects
        alignment.append(Alignment.from_strings(i,j))

    return alignment


def collapse_tokens(doc) -> list:

    # The function reconstructs the tokenization done by spacy
    # on the corpus to align it to the coNLL tokenization
    # :param doc: a tokenized sentence
    # :return: the list of reconstructed tokens and relative tags

    # Initialize needed objects
    collapsed_tokens = []
    tok = ""
    tag = ""
    flag = True

    # Iterate over the length of the input doc
    for i, token in enumerate(doc):
        # Store the current token in an accumulator
        tok += token.text

        # Reconstruct the tag according to the IOB type
        if flag:
            if token.ent_iob_ == "O":
                tag = token.ent_iob_
            else:
                tag = token.ent_iob_ + "-" + labels[token.ent_type_]
            flag = False

        # When we encounter a whitespace...
        if token.whitespace_ == " ":
            # ...we append the collapsed token and relative tag to a list
            collapsed_tokens.append((tok, tag))

            # Reset the objects controlling the loop
            tok = ""
            flag = True

        # When we reach the penultimate token, we append
        # the reconstructed token and relative tag
        if i == len(doc)-1:
            collapsed_tokens.append((tok, tag))

    return collapsed_tokens



def collapse_tokens_alternative(corpus) -> list:

    # The function attempts to reconstruct the coNLL tokenization by
    # collapsing the tokens that have been separated by spacy
    # :param corpus: corpus in coNLL format 
    # :return: the list of reconstructed tokens and relative tags

    # Retrieve or initialize the needed objects
    list0 = get_spacy_tags_clean(corpus)
    wh = get_whitespaces(corpus)
    list2 = list()
    list4 = list()

    # Iterate over the length of the whitespaces list
    for i in range(len(wh)):
        j = 0
        while j < len(wh[i]):

            # Initialize an accumulator to store the collapsed token
            acc = list0[i][j][0]
        
            # If we encounter a whitespace, we simply append the token
            if wh[i][j] == True:
                list2.append(list0[i][j])
                j += 1
        
            # If we reach the end of the sentence, we append the token
            elif j == len(wh[i])-1:
                list2.append(list0[i][j])
                j += 1
        
            # If we don't have a whitespace, we collapse tokens
            elif wh[i][j] == False:
                # Store the first encountered IOB tag
                tag = list0[i][j][1]
                # As long as we do not reach a whitespace, we keep collapsing
                while wh[i][j] == False and j < len(wh[i])-1:
                    if spacy_plus_tags_clean[i][j+1][0] in ",.:;'":
                        # Break the loop when reaching punctuation
                        break

                    # Store the collapsed token
                    acc += list0[i][j+1][0]
                    j += 1
                # Store the collapsed token and relative tag
                list2.append((acc, tag))
                j += 1 
        # Append the object to another list for suitable format
        list4.append(list2)
        list2 = list()
    
    return list4 



def get_stats(corpus):

    # The function computes the chunk-level performance
    # as given by the conll.evaluate() function 
    # :param corpus: corpus in coNLL format
    # :return: evaluation of chunk-level performance

    # Initialize empty list for storing references
    ref = list()
    base = clean_sents(corpus)
    phrases = [sent[0] for sent in base]
    
    # Call the collapse_tokens(doc) function and append the results
    for sent in phrases:
        spacy_doc = nlp(sent)
        ref.append(collapse_tokens(spacy_doc))

    # Retrieve the original corpus
    hyp = extract_tags_clean(corpus)

    # Call the evaluation function in conll.py
    return conll.evaluate(ref, hyp)



def accuracy_token_level(corpus):

    # The function computes the token-level accuracy
    # as given by the sklearn classification_report() function
    # :param corpus: corpus in coNLL format 
    # :return: classification report of sklearn

    #corpus_ner = get_spacy_tags_clean(corpus)
    base = clean_sents(corpus)
    phrases = [sent[0] for sent in base]
    words_plus_tags_clean = extract_tags_clean(corpus)

    # Initialize needed objects
    tot_count = list()
    pos_count = list()

    # Iterate over the sentences of the corpus in string format
    for i, sent in enumerate(words_plus_tags_clean):
        
        # Tokenize each sentence with spacy
        spacy_doc = nlp(phrases[i])
        # Reconstruct coNLL tokenization 
        tokens = collapse_tokens(spacy_doc)

        # Iterate over the tokens of each sentence
        for j in range(len(tokens)):
            # Update the list of coNLL tags
            tot_count.append(sent[j][1])
            # Update the list of spacy tags
            pos_count.append(tokens[j][1])

    # Call the sklean function to compute accuracy and other 
    # relevant statistics on the obtained objects
    return classification_report(tot_count, pos_count)



def get_chunks_ent(corpus) -> dict:

    # The function performs grouping of noun chunks entities 
    # and counts their occurrrences throughout the corpus
    # :param corpus: corpus in coNLL format 
    # :return: a dictionary containing occurrences

    # Retrieve sentences in a suitable format for analysis
    crp = clean_sents(corpus)
    phrases = [sent[0] for sent in crp]

    # Re-load the language model
    nlp = spacy.load('en_core_web_sm')

    # Obtain the references
    ref = get_spacy_tags_clean(corpus)

    # Initialize accumulation lists
    noun_chunks_ents = []
    inside = []
    
    # Iterate over the length of the corpus
    for i in range(len(phrases)): 

        # Tokenize each sentence 
        doc = nlp(phrases[i])   

        # Retrieve noun chunks of each sentence
        for chunk in doc.noun_chunks:

            # Store them into the lists
            inside.append(str(chunk).split())
        noun_chunks_ents.append(inside)
        inside = list()

    # Prepare the needed object for the count of entities
    temp_ext = []
    temp_out = []
    temp_in = []

    # Iterate over the length of the sentences (chunked)
    for s, sent in enumerate(noun_chunks_ents):
        # print(s)  ## optional counter to keep track of the process

        # Iterate over the number of chunks of each sentence
        for n in range(len(sent)):
            # Iterate over the number of elements of each chunk
            for k in range(len(sent[n])):
                # Iterate over the number of sentences of the reference
                for i in range(len(ref)):
                    # Iterate over the length of each sentence
                    for j in range(len(ref[i])):
                        # Check whether the current chunk element is equal to 
                        # the current token from the reference
                        if str(sent[n][k]) == ref[i][j][0]:
                            # If so, store its NE into a list and break loop
                            temp_in.append(ref[i][j][1])
                            break
                    # Break loop when you reach end of sentence
                    break
            # Store collected objects into lists in suitable format
            temp_out.append(temp_in)
            temp_in = list()
        temp_ext.append(temp_out)
        ref = ref[1:]
        temp_out = list()

    # Begin the computation of groups of entities
    stats = dict()

    # Iterate over the chunked sentences
    for ents in temp_ext:
        # Iterate over the number of chunks in each sentence
        for i in ents:
            # Construct the keys of the dictionary by collecting
            # the unique combinations of NE of each chunk...
            index = list()
            for u in np.unique(i):
                index.append(str(u))
            # ...then initialize the keys
            stats[tuple(index)] = 0
    # Now iterate over the entities of the chunks...
    for ents in temp_ext:
        for i in ents:
            index = list()
            for u in np.unique(i):
                index.append(str(u))
            # ...and update the values of the dictionary
            stats[tuple(index)] +=1
    
    return(stats)



def span_compound(doc):
    
    # The function fixes eventual segmentation errors in
    # the corpus by using the compound dependency relation
    # :param doc: a tokenized sentence
    # :return: a doc object with fixed segmentation errors

    # Initialize a controlling parameter 
    flag = True

    # Iterate until the parameter changes
    while flag:
        flag = False

        # Iterate over the entities of the doc object
        for ent in doc.ents:
            # Iterate over the tokens of each entity
            for token in ent:
                # Iterate over the children of each token
                for child in token.children:
                    # If the dependency relation of the token is compound...
                    if child.dep_ == "compound": 
                        # If its IOB tag is 'O'...   
                        if child.ent_iob_ == 'O':
                            # If the entity it belongs to is not located at 
                            # the very beginning of the sentence...
                            if ent.start-1 >= 0:
                                # If the child corresponds to the one before the start...
                                if child == doc[ent.start-1]:
                                    # ...create a new span which extends to comprise it
                                    new_span = Span(doc, ent.start-1, ent.end, label=ent.label_)
                                    # Fix the entity by using doc.set_ents of spacy
                                    doc.set_ents([new_span], default="unmodified")
                                    # Change the controlling parameter 
                                    # and continue the loop
                                    flag = True
                    # The same procedure applies for entities located at
                    # the end of the doc object
                    if child.dep_ == "compound":     
                        if child.ent_iob_ == 'O':
                            if ent.end+1 < len(doc):
                                if child == doc[ent.end+1]:
                                    # Create the new extended span
                                    new_span = Span(doc, ent.start, ent.end+1, label=ent.label_)
                                    # Fix the entity through spacy
                                    doc.set_ents([new_span], default="unmodified")
                                    flag = True
        
        # Store results in a final doc object and return it
        final_spanned_doc = doc

    return final_spanned_doc





## TESTING ----------------------------------------------------------

## EXERCISE 0
print(accuracy_token_level(test))
pprint(dict(get_stats(test)))


## EXERCISE 1
stats = get_chunks_ent(test)
for key, value in stats.items():
        print('{}\t{}'.format(key, value))


## EXERCISE 2
spanned_corpus = list()
ref = extract_tags_clean(corpus)
for sent in ref:
    doc = nlp(sent)
    spanned_corpus.append(span_compound(doc))
