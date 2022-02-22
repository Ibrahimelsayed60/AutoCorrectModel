### First Part

#### Import libraries
import re
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

def process_data(file):
    with open(file) as input:
        data = input.read().strip()
    data_lowercase = data.lower()
    words = re.findall(r'\w+', data_lowercase)
    return words


## Unit test of process_data function

words = process_data('shakespeare.txt')
vocab = set(words)
print(f'The first ten words in the text are: \n{words[0:10]}')
print(f"There are {len(vocab)} unique words in the vocabulary. ")

def get_count(words_list):
    word_count = dict()
    for w in words_list:
        if w in word_count:
            word_count[w] += 1
        else:
            word_count[w] = 1
    return word_count

## Unit test of get_count function
word_count_dict = get_count(words)
print(f"There are {len(word_count_dict)} key values pairs")
print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")


def get_probs(word_dict):
    word_proba = dict()
    m = sum(word_dict.values())
    for w in word_dict.keys():
        word_proba[w] = word_dict.get(w,0) / m
    
    return word_proba

# Unit test for get_probs function
probs = get_probs(word_count_dict)
print(f"Length of proba is {len(probs)}")
print(f"P('thee') is {probs['thee']:.4f}")


### Second part: String Manipulations:

# Delete function
def delete_letter(word, verbose = False):
    split_list = []
    delete_list = []

    for i in range(len(word)):
        split_list.append([word[:i], word[i:]])
    
    for L, R in split_list:
        if R:
            delete_list.append(L+R[1:])
    if verbose: 
        print(f"input word {word}, \nsplit_list = {split_list}, \ndelete_list = {delete_list}")
    
    return delete_list

# Unit test for delete_letter function
delete_word_list = delete_letter(word='cans', verbose = True)

## To get the solit_list = [['', 'cans'], ['c', 'ans'], ['ca', 'ns'], ['can', 's'], ['cans', '']], we replace (len(word)) by (len(word)+1)
# Second unit test for delete_letter function 
print(f"Number of outputs of delete_letter('at') is {len(delete_letter('at'))}")


def switch_letter(word, verbose = False):
    switch_list = []
    split_list = []

    for i in range(len(word)):
        split_list.append([word[:i] , word[i:]])
    for L,R in split_list:
        if len(R) >= 2:
            switch_list.append( L + R[1] + R[0] + R[2:])

    if verbose: print(f"Input word = {word} \nSplit_list = {split_list} \nSwitch_list = {switch_list}")

    return switch_list

#unit Test for switch_letter function
switch_word_list = switch_letter(word="eta",verbose=True)
# test # 2
print(f"Number of outputs of switch_letter('at') is {len(switch_letter('at'))}")


def replace_letter(word, verbose = False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_list = []
    split_list = []
    for i in range(len(word)):
        split_list.append([word[:i] , word[i:]])
    
    for L, R in split_list:
        if(len(R) >= 1):
            for l in letters:
                replace_list.append(L + l + R[1:])

    replace_set = set(replace_list)
    replace_set.remove(word)
    replace_list = sorted(list(replace_set))

    if verbose: print(f"Input word = {word} \nSplit_list = {split_list} \nReplace_list = {replace_list}")

    return replace_set

#Unit test for rplace_letter function
replace_word_list = replace_letter(word = 'can', verbose = True)
print(f"Number of outputs of switch_letter('at') is {len(switch_letter('at'))}")


def insert_letter(word, verbose = False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_list = []
    split_list = []
    for i in range(len(word)+1):
        split_list.append([word[:i], word[i:]])
    for L, R in split_list:
        for l in letters:
            insert_list.append(L + l + R)
    
    if verbose: print(f"Input word: {word} \nSplit_list: {split_list} \nInsert_list: {insert_list}")

    return insert_list

insert_word_list = insert_letter('at',True)
print(f"Number of Strings output by inser_letter('at') is {len(insert_word_list)}")

# Third Part: Combining the edits
def edit_one_letter(word, allow_switches = True):
    edit_one_set = set()
    if(allow_switches):
        edit_one_set = set([*replace_letter(word), *insert_letter(word),*delete_letter(word),*switch_letter(word)])
    else:
        edit_one_set = set([*replace_letter(word),*insert_letter(word),*delete_letter(word)])

    return edit_one_set

# Unit Test for edit_one_letter function
tmp_word = "at"
tmp_edit_one_set = edit_one_letter(tmp_word)
# turn this into a list to sort it, in order to view it
tmp_edit_one_l = sorted(list(tmp_edit_one_set))

print(f"input word {tmp_word} \nedit_one_l \n{tmp_edit_one_l}\n")
print(f"The type of the returned object should be a set {type(tmp_edit_one_set)}")
print(f"Number of outputs from edit_one_letter('at') is {len(edit_one_letter('at'))}")

# Part 3.2: Edit two letters:
def edit_two_letters(word,allow_switches = True):
    edit_two_set = set()
    edit_one = edit_one_letter(word,allow_switches = True)
    # if(allow_switches ):
        # edit_two_set = set([set.union(edit_one_letter(word, True)) for word in edit_one if word])
    for w in edit_one:
        if w:
            edit_two = edit_one_letter(w,allow_switches)
            edit_two_set.update(edit_two)
    else:
        edit_two_set = set([set.union(edit_one_letter(word)) for word in edit_one if word])
            
    return edit_two_set

#Unit test for edit_two_letters function:
tmp_edit_two_set = edit_two_letters("a")
tmp_edit_two_l = sorted(list(tmp_edit_two_set))
print(f"Number of strings with edit distance of two: {len(tmp_edit_two_l)}")
print(f"First 10 strings {tmp_edit_two_l[:10]}")
print(f"Last 10 strings {tmp_edit_two_l[-10:]}")
print(f"The data type of the returned object should be a set {type(tmp_edit_two_set)}")
print(f"Number of strings that are 2 edit distances from 'at' is {len(edit_two_letters('at'))}")


# Part 3.3: Suggest spelling suggestions

# example of logical operation on lists or sets
print( [] and ["a","b"] )
print( [] or ["a","b"] )
#example of Short circuit behavior
val1 =  ["Most","Likely"] or ["Less","so"] or ["least","of","all"]  # selects first, does not evalute remainder
print(val1)
val2 =  [] or [] or ["least","of","all"] # continues evaluation until there is a non-empty list
print(val2)

def get_corrections(word, probs, vocab, n=2, verbose = False):

    suggestions = []
    n_best = []

    suggestions = list((word in vocab and word) or edit_one_letter(word).intersection(vocab))
    n_best = [[s, probs[s]] for s in list(reversed(suggestions))] 

    if verbose: print("suggestions = ", suggestions)

    return n_best

# Test your implementation - feel free to try other words in my word
my_word = 'dys' 
tmp_corrections = get_corrections(my_word, probs, vocab, 2, verbose=True)
for i, word_prob in enumerate(tmp_corrections):
    print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")

# CODE REVIEW COMMENT: using "tmp_corrections" insteads of "cors". "cors" is not defined
print(f"data type of corrections {type(tmp_corrections)}")

#Part 4: Minimum Edit distance
#Part 4.1: Dynamic Programming

def min_edit_distance(source, target, ins_cost = 1, del_cost = 1, rep_cost = 2):
    '''
    Input: 
        source: a string corresponding to the string you are starting with
        target: a string corresponding to the string you want to end with
        ins_cost: an integer setting the insert cost
        del_cost: an integer setting the delete cost
        rep_cost: an integer setting the replace cost
    Output:
        D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        med: the minimum edit distance (med) required to convert the source string to the target 
    '''

     # use deletion and insert cost as  1
    m = len(source) 
    n = len(target) 
    #initialize cost matrix with zeros and dimensions (m+1,n+1) 
    D = np.zeros((m+1, n+1), dtype=int) 
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    
    # Fill in column 0, from row 1 to row m, both inclusive
    for row in range(1,m+1): # Replace None with the proper range
        D[row,0] = D[row-1,0] + del_cost
        
    # Fill in row 0, for all columns from 1 to n, both inclusive
    for col in range(1,n+1): # Replace None with the proper range
        D[0,col] = D[0,col-1] + ins_cost
        
    # Loop through row 1 to row m, both inclusive
    for row in range(1,m+1): 
        
        # Loop through column 1 to column n, both inclusive
        for col in range(1,n+1):
            
            # Intialize r_cost to the 'replace' cost that is passed into this function
            r_cost = rep_cost
            
            # Check to see if source character at the previous row
            # matches the target character at the previous column, 
            if source[row-1] == target[col-1]:
                # Update the replacement cost to 0 if source and target are the same
                r_cost = 0
                
            # Update the cost at row, col based on previous entries in the cost matrix
            # Refer to the equation calculate for D[i,j] (the minimum of three calculated costs)
            D[row,col] = min([D[row-1,col]+del_cost, D[row,col-1]+ins_cost, D[row-1,col-1]+r_cost])
          
    # Set the minimum edit distance with the cost found at row m, column n
    med = D[m,n]

    return D, med

#DO NOT MODIFY THIS CELL
# testing your implementation 
source =  'play'
target = 'stay'
matrix, min_edits = min_edit_distance(source, target)
print("minimum edits: ",min_edits, "\n")
idx = list('#' + source)
cols = list('#' + target)
df = pd.DataFrame(matrix, index=idx, columns= cols)
print(df)


#DO NOT MODIFY THIS CELL
# testing your implementation 
source =  'eer'
target = 'near'
matrix, min_edits = min_edit_distance(source, target)
print("minimum edits: ",min_edits, "\n")
idx = list(source)
idx.insert(0, '#')
cols = list(target)
cols.insert(0, '#')
df = pd.DataFrame(matrix, index=idx, columns= cols)
print(df)

source = "eer"
targets = edit_one_letter(source,allow_switches = False)  #disable switches since min_edit_distance does not include them
for t in targets:
    _, min_edits = min_edit_distance(source, t,1,1,1)  # set ins, del, sub costs all to one
    if min_edits != 1: print(source, t, min_edits)


source = "eer"
targets = edit_two_letters(source,allow_switches = False) #disable switches since min_edit_distance does not include them
for t in targets:
    _, min_edits = min_edit_distance(source, t,1,1,1)  # set ins, del, sub costs all to one
    if min_edits != 2 and min_edits != 1: print(source, t, min_edits)


