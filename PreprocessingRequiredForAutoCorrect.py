### Import liraries required
import re ## regular expression library
from collections import Counter

from sqlalchemy import intersect
from sympy import intersection
# import matplotlib.pyplot as plt


text = "red pink pink blue blue yellow ORANGE BLUE BLUE PINK"

## preprocessing
text_lowercase = text.lower()

# some regex to tokenize the string to words and return them in a list
# print(text_lowercase)
words = re.findall(r'\w+', text_lowercase)
words_split = text_lowercase.split(" ")

# Make a set of distinct words from the text

vocab = set(words)

# Creat a dictionary contains the words and count of each word
count_a = dict()

for w in words:
    if w in count_a:
        count_a[w] += 1
    else:
        count_a[w] = 1

# print(count_a)
# Creat a dictionary contains the words and count of each word using Counter

counts_b = dict()
counts_b = Counter(words)

print(counts_b)

import matplotlib.pyplot as plt

# barchart of sorted word counts
d = {'blue': counts_b['blue'], 'pink': counts_b['pink'], 'red': counts_b['red'], 'yellow': counts_b['yellow'], 'orange': counts_b['orange']}
plt.bar(range(len(d)), list(d.values()), align='center', color=d.keys())
_ = plt.xticks(range(len(d)), list(d.keys()))
# plt.show()
### Exercise 2

word = "dearz"

#Split the word
split_a = []
for i in range(len(word)+1):
    split_a.append([word[:i],word[i:]])
for i in split_a:
    print(i)

# same split, done using a list comprehension
split_b = [ (word[:i],word[i:]) for i in range(len(word)+1)]

for i in split_b:
    print(i)

## Edit: Delete
splits = split_a
deletes = []
for L,R in splits:
    if R:
        print(L + R[1:],' <-- delete', R[0])

deletes = [L + R[1:] for L,R in splits]

print("delete list: ",deletes)

vocab = ['dean','deer','dear','fries','and','coke']
edits = list(deletes)
print(edits)
candidates = []
# Get the intersection between two lists
candidates = [ i for i in vocab for j in edits if i==j]
print(candidates)













