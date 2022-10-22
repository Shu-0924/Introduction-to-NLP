# Author: Bing-Shu Wu
# Student ID: 109550074
# HW ID: HW1
import spacy
import csv

nlp = spacy.load("en_core_web_sm")
fin = open('dataset.csv')
csv_reader = csv.reader(fin)
ground_truth = []
result = []


for row in csv_reader:
    ground_truth.append(int(row[0]))
    result.append(0)
    doc = nlp(row[1])

    # Check every word in 'V' is in the sentence
    V_start_end_idx = [None, None]
    verb_candidates = "-".join(row[3].strip().split()).split("-")
    all_words = "-".join(row[1].strip().split()).split("-")
    if sum([w not in all_words for w in verb_candidates]) > 0:
        continue

    # Find the doc's index of all verb in 'V' column in .csv
    i = 0
    verb_idxs = []
    for j, token in enumerate(doc):
        if i < len(verb_candidates) and token.text == verb_candidates[i]:
            if token.pos_ == 'VERB' or token.pos_ == 'AUX':
                verb_idxs.append(j)
            i += 1
    for j, token in enumerate(doc):
        if token.text == verb_candidates[0]:
            V_start_end_idx[0] = j
        if token.text == verb_candidates[-1]:
            V_start_end_idx[1] = j

    # Check the (S, V, O) tuple in .csv is correct or not by looking at every verb in 'V' column
    for verb_idx in verb_idxs:
        # find 'S'
        S_list = [None]
        i = V_start_end_idx[0] - 1
        punct_clause = (doc[verb_idx].dep_ == 'advcl' and doc[verb_idx - 1].pos_ == 'PUNCT')
        while i >= 0:
            if ('subj' in doc[i].dep_ and doc[i].head.i == verb_idx) or \
                    ('subj' in doc[i].dep_ and doc[i].head.i == doc[verb_idx].head.i and punct_clause):
                if doc[i].text in ['which', 'who', 'that', 'where']:
                    i = doc[verb_idx].head.i
                subtree = list(doc[i].subtree)
                start = subtree[0].i
                S_list[0] = doc[start:i+1].text if doc[start:i+1].text is not None else None
                break
            i -= 1

        # find 'O'
        O_list = [None]
        i = V_start_end_idx[1] + 1
        while i < len(doc):
            if 'obj' in doc[i].dep_ or 'attr' in doc[i].dep_:
                layer_1 = doc[i].head.i
                layer_2 = doc[layer_1].head.i
                layer_3 = doc[layer_2].head.i
                if layer_1 == verb_idx or (doc[layer_1].pos_ == 'ADP' and layer_2 == verb_idx) or \
                        (doc[layer_1].pos_ == 'ADP' and doc[layer_2].pos_ == 'NOUN' and layer_3 == verb_idx):
                    subtree = list(doc[i].subtree)
                    start = subtree[0].i
                    # end = subtree[-1].i + 1 if doc[i+1].pos_ != "PUNCT" and doc[i+1].text not in \
                    #    ['when', 'who', 'that', 'which', 'what', 'where', 'with', 'to', 'at'] else i+1
                    end = subtree[-1].i+1
                    O_list[0] = doc[start:end].text if doc[start:end].text is not None else None
                    break
            i += 1

        # if S_list[0] is not None and S_list[0] in row[2] and O_list[0] is not None and O_list[0] in row[4]:
        if S_list[0] is not None and (S_list[0] in row[2] or row[2] in S_list[0]) and O_list[0] is not None and (O_list[0] in row[4] or row[4] in O_list[0]):
            result[-1] = 1
        # print(f"row No.{len(result)+1} with {doc[verb_idx].pos_} '{doc[verb_idx]}': ")
        # print(f"(S, V, O) -> ({S_list[0]}, {doc[verb_idx]}, {O_list[0]}), result: {result[-1]}\n")


fin.close()
with open('result.csv', 'w', newline='') as fout:
    csv_writer = csv.writer(fout)
    csv_writer.writerow(['index', 'T/F'])
    for i in range(len(result)):
        csv_writer.writerow([i, result[i]])


"""
# Compute accuracy rate ('example_with_answer.csv' ONLY)
accuracy = 0
false_positive = 0
false_negative = 0
for y, t in zip(result, ground_truth):
    if y-t == 0:
        accuracy += 1
    elif y == 1:
        false_positive += 1
    else:
        false_negative += 1
        
print(f"Accuracy rate: {100*float(accuracy)/len(result)}%")
print(f"False positive: {100*float(false_positive)/len(result)}%")
print(f"False negative: {100*float(false_negative)/len(result)}%")
"""


"""
# Debug tool
doc = nlp("Rhodes discovered he had cancer last October after he felt a sharp pain in his right leg .")
for token in doc:
    print(f"{token.text}\t{token.pos_}\t{token.dep_}\t{token.head.text}\t{[a for a in token.subtree]}")
"""
