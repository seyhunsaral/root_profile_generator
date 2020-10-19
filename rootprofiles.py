import itertools
import math
import numpy as np
import timeit
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import string
from collections import Counter

def accel_asc(n):
# Generating partitions of n
# Kelleher, Jerome, and Barry O'Sullivan. "Generating all partitions: a comparison of two encodings." arXiv preprint arXiv:0909.2331 (2009).
    a = [0 for i in range(n + 1)]
    k = 1
    a[0] = 0
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2*x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]



def balls_to_boxes(n, b):
    """ n balls b boxes problem. Returns all possible distributions in an iterable"""
    # I found the code exammpe https://stackoverflow.com/a/37712597/1819625 by Jared Gougen
    partition_array = np.empty((math.comb(n+b-1, b-1), b), dtype=np.int16)
    masks = np.identity(b, dtype=np.int16)
    for i, c in enumerate(itertools.combinations_with_replacement(masks, n)): 
        partition_array[i,:] = sum(c)
    return partition_array

def delete_if_first_not_max(np_array):
    """ takes a numpy array of arrays (matrix). if the first element of the vector is not the maximal element, it deletes the vector"""
    # I used this function because the way I create normalized vectors are not 

    max_array = np.amax(np_array, axis=1)
    return np_array[np_array[:,0] >= max_array]



##############################################
##   Generating Normalized Vectors          ##
##############################################

## METHOD 1 - Direct method

# In this method, I create empty boxes for possible profiles and get
# possible partitions of number o candidates. Given we have a maximum
# member on a partition, we create the first column with the maximum
# element, and the rest from the remainng partitions.


def generate_normalized_direct(num_voters, num_candidates, verbose = False):
    # Note: for technical reasons the output is not ordered
    # The maximal element is on the first column decreasing but the rest increases
    # We can of course sort them but this is extra calculation
    num_boxes = math.factorial(num_candidates)
    partitions = list(accel_asc(num_voters))

    vectors = np.empty((0,num_boxes), dtype=np.int16)
    length_partitions = len(partitions)
    if verbose:
        print("length of partitions", length_partitions)
    for ind, my_partition in enumerate(partitions):
        if verbose:
            print(ind, "/", length_partitions)

        # Take the last element (which is the maximum for accel_asc function)
        current_v1 = my_partition.pop()

        # Calculate how many zeros to permute with the partitioned values
        len_zeros = num_boxes - len(my_partition) -1
        if len_zeros >= 0:
            right_side = [0] * len_zeros + my_partition # vector to permute


            # This is the permutation function ignores repetitions
            right_side_array = np.asarray(list(multiset_permutations(right_side)))
            left_side_array = np.full((np.shape(right_side_array)[0],1),current_v1)

            full_array = np.hstack((left_side_array, right_side_array))
            vectors = np.vstack((vectors,full_array))

    return vectors



## METHOD 2 - Generates anonymous vector set first

# In this method, I create all anonymous vectors first and then delete
# with the first element is not maximum. It is obviously "inefficient"
# but surprisingly in some cases it is faster as we use itertools for
# generation. 

def generate_normalized_indirect(number_of_voters, number_of_candidates):
    number_of_balls = number_of_voters
    number_of_boxes = math.factorial(number_of_candidates)
    all_anonymous_vectors = balls_to_boxes(number_of_balls, number_of_boxes)
    normalized_vectors = delete_if_first_not_max(all_anonymous_vectors)
    return normalized_vectors



## METHOD 3 - Semi-direct: Generates everything except first column 

# It is almost like a mixture of Method 1 and 2. Instead of getting
# full partitions of voters, we just take the number of balls for the
# first column and generate all possible situations for the rest.In
# some profiles, v1 is not maximum so we delete those.

def generate_normalized_semidirect(number_of_voters, number_of_candidates):
    number_of_balls = number_of_voters
    number_of_boxes = math.factorial(number_of_candidates)
    normalized_efficient = np.empty((0,number_of_boxes) ,dtype='int')
    initial = [np.concatenate((np.array([number_of_balls]), np.zeros(number_of_boxes-1, dtype=np.int16)))]
    normalized_efficient = np.append(normalized_efficient, initial, axis = 0)

    for balls_right in range(1,number_of_balls):
        balls_left = number_of_balls - balls_right
        current_right= balls_to_boxes(balls_right, number_of_boxes-1)
        current_left = np.full((np.shape(current_right)[0],1), balls_left)
    #    left_column[:,:-1] = current_array
        current_vector = np.concatenate((current_left,current_right), axis=1)
        normalized_efficient = np.append(normalized_efficient, current_vector, axis = 0)

    normalized = delete_if_first_not_max(normalized_efficient)
    return normalized



##  This is just to calculate how many combs needed for semi-direct, for information only

def normalized_semidirect_num_vectors(number_of_voters, number_of_candidates):
    number_of_balls = number_of_voters
    number_of_boxes = math.factorial(number_of_candidates)
    num_vectors = 1

    for balls_right in range(1,number_of_balls):
        balls_left = number_of_balls - balls_right
        current= math.comb(balls_right + number_of_boxes-2, balls_right)
        num_vectors += current
    return num_vectors


##### benchmark 

## UNCOMMENT BELOW TO SEE THE TESTS
#--------------- tests start ---------------

#test_parameters = ["(10,2)", "(20,2)", "(50,2)", "(3,3)", "(10,3)", "(20,3)", "(4,4)", "(6,4)"]
#
#number_of_times = 10
#
#for test_parameter in test_parameters:
#    print("--")
#    print(test_parameter)
#    print("number_of_repetitions: ", number_of_times)
#    print("Direct method:     ", timeit.timeit("generate_normalized_direct" + test_parameter, globals = globals(), number = number_of_times))
#    print("Indirect method:   ", timeit.timeit("generate_normalized_indirect" + test_parameter, globals = globals(), number = number_of_times))
#    print("Semidirect method: ", timeit.timeit("generate_normalized_semidirect" + test_parameter, globals = globals(), number = number_of_times))
#
##-----------------tests end ---------------------
###    (10,2)
###    number_of_repetitions:  10
###    Direct method:      0.0018116440005542245
### *  Indirect method:    0.0010369669998908648
###    Semidirect method:  0.002164452000215533
###    --
###    (20,2)
###    number_of_repetitions:  10
### *  Direct method:      0.007081401001414633
###    Indirect method:    0.008951568999691517
###    Semidirect method:  0.011230537998926593
###    --
###    (50,2)
###    number_of_repetitions:  10
###    Direct method:      1.0699581059998309
###    Indirect method:    0.02324520200090774
### *  Semidirect method:  0.014575720000721049
###    --
###    (3,3)
###    number_of_repetitions:  10
###    Direct method:      0.0037149619984120363
###    Indirect method:    0.0013967100003355881
### *  Semidirect method:  0.0009716330005176133
###    --
###    (10,3)
###    number_of_repetitions:  10
### *  Direct method:      0.0669338739990053
###    Indirect method:    0.16643244800070534
###    Semidirect method:  0.09733788599987747
###    --
###    (20,3)
###    number_of_repetitions:  10
### *  Direct method:      0.5515356460000476
###    Indirect method:    5.224526124000477
###    Semidirect method:  3.325670046000596
###    --
###    (4,4)
###    number_of_repetitions:  10
###    Direct method:      0.16393854600028135
###    Indirect method:    0.4688242070005799
### *  Semidirect method:  0.06143047200021101
###    --
###    (6,4)
###    number_of_repetitions:  10
###    Direct method:      3.63905186699958
###    Indirect method:    16.34116097200058
### *  Semidirect method:  3.2118038939988764


def convert_condensed(voters):
    """
    This functions gets a list of voters and convert in each voter preferences in condensed string form. 

    Example:
    list_of_voters = [["a","b","c","d"],["b","a","c","d"]]

    convert_condensed(list_of_voters)
    ## Output: ['abcd', 'bacd']
    """
    return np.asarray(["".join(voter) for voter in voters])

def convert_extended(voters):
    """
    This functions gets a list of voters in condensed form and convert extended string form. 

    Example:
    list_of_voters = [["a","b","c","d"],["b","a","c","d"]]

    convert_condensed(list_of_voters)
    ## Output: ['abcd', 'bacd']
    """
    num_voters = len(voters)
    num_candidates = len(voters[0])
    voters = np.asarray(voters)
    new_voters = np.empty((num_voters, num_candidates), dtype='str')
    for index, voter in enumerate(voters):
        new_voters[index,] = list(voter)
    return new_voters

def all_perms(candidate_list, as_nparray=True):
    perms=list(itertools.permutations(candidate_list))
    
    if as_nparray:
        perms = np.array(perms)
    
    return perms


def all_possible_prefs(candidates, concentrate=False):
    # Gives the possible preferences given the action list
    perms = all_perms(candidates)

    if concentrate:
        perms = convert_condensed(perms)

    return perms



def get_alphabet_firstn(n):
    """
    get_alphabet_firstn(n)

    Returns the firtst n letters of the alphabet

    Parameters
    ----------
    n: string
        number of letters

    Returns 
    -------
    list 

    Example
    -------
    get_alphabet_firstn(3) # returns ['a','b','c']
    """
    return list(string.ascii_lowercase[0:n])

def summarize_voters(voters):
    return dict(Counter(convert_condensed(voters)))



def comp_vector_to_profile(comp_vector, possible_prefs):
    take_index = [[index] * value for index, value in enumerate(list(comp_vector))]
    take_index_flatten = list(itertools.chain.from_iterable(take_index))
    return np.take(possible_prefs, take_index_flatten, axis=0)


def profile_to_comp_vector(voters, possible_prefs):
    condensed_voters = Counter(convert_condensed(voters))
    condensed_prefs = convert_condensed(possible_prefs)
    num_candidates = len(possible_prefs[0])
    composition_vector = np.zeros(math.factorial(num_candidates),  dtype=np.int16)

    for key, value in condensed_voters.items():
        composition_index = np.where(condensed_prefs == key)[0][0]
        composition_vector[composition_index] = value
    return composition_vector


def make_dictionary(pref1, pref2):
    return dict(zip(pref1, pref2))
    


def rename_candidates(voters, dictionary):
    new_voters = np.copy(voters)
    for i in range(np.shape(voters)[0]):
        for j in range(np.shape(voters)[1]):
            new_voters[i,j] = dictionary[voters[i,j]]
    return new_voters


# Renaming can be done with a dictionary
# for instance  
#     candidate_map = {"a":"b", "b":"c", "c":"a"}

# We can generate such a map from two rankings 
#     another_candidate_map = make_dictionary(possible_prefs[0], possible_prefs[3])

# rename_candidates(new_profile, make_dictionary())



def map_vector_entries(vector, index_from, index_to, possible_prefs):
# This function gets a composition vector, takes renames candidates in a way that one column maps to the other then gives the outcome as a vector. Just like the last example in the preseantation where I eliminated some profiles.
    # generates the profile 
    profile_from_vector =  comp_vector_to_profile(vector, possible_prefs)
    # creates a dictionary for renaming
    mapping = make_dictionary(possible_prefs[index_from], possible_prefs[index_to])
    # renames
    renamed_profile = rename_candidates(profile_from_vector, mapping)
    # converts back to vector
    vector_from_profile = profile_to_comp_vector(renamed_profile, possible_prefs)
    return vector_from_profile


def get_root_from_normalized_vector(normalized, possible_prefs):
# Takes a normalized vector array and if there are more than 1 max
# element, it maps other maximal elments to first column. If it
# coincides with another profile, it deletes that profile.
    normalized_copy = np.copy(normalized) # take a copy

    current_length = len(normalized_copy) 
    i = 0
    # I chose while loop over for in order to delete on the fly
    while i < current_length:
        row = normalized_copy[i]
        max_in_row = np.amax(row)
        max_rows = np.where(row == max_in_row)
        # check if there are multiple maximums in a row
        if np.shape(max_rows)[1] > 1:
            cols_to_map = max_rows[0][1:]  # take all except col 1
            for col in cols_to_map:
                current_mapped = map_vector_entries(row, col, 0, possible_prefs)
                if not (current_mapped == row).all():

                    index_to_delete = np.where((normalized_copy == current_mapped).all(axis=1))

                    if index_to_delete:
                        index_to_delete = index_to_delete[0]
                        normalized_copy = np.delete(normalized_copy, index_to_delete, axis=0)
                        current_length = len(normalized_copy)
        i += 1

    return normalized_copy


def generate_roots(number_of_voters, number_of_candidates, method="semidirect"):
    candidates = get_alphabet_firstn(number_of_candidates)
    possible_prefs = all_possible_prefs(candidates)

    if method == "direct":
        normalized_vectors = generate_normalized_direct(number_of_voters, number_of_candidates)

    if method == "indirect":
        normalized_vectors = generate_normalized_indirect(number_of_voters, number_of_candidates)

    if method == "semidirect":
        normalized_vectors = generate_normalized_semidirect(number_of_voters, number_of_candidates)

    roots = get_root_from_normalized_vector(normalized_vectors, possible_prefs)

    return roots

# End of functions
#===============================================================================
#=====================  Generation                   ===========================
#===============================================================================

number_of_candidates = 4
number_of_voters = 4
print_profiles =  True # or it will just show the numbers 

# Generate Normalized (for demonstration only, it is created by generate_roots function)
normalized = generate_normalized_semidirect(number_of_candidates, number_of_voters)

# Roots
# If you are running calculations, it is better comment out normalized genration, or to generate roots from normalized vector direclty by "get_root_from_normalized_vector" function in order not to calculate the same thing twise.
roots = generate_roots(number_of_voters, number_of_candidates)

if print_profiles:
    print("\n" * 2)
    print('roots in vector form')
    print('-' * 10)
    print(roots)
=======
roots = get_root_from_normalized_vector(normalized_vectors, possible_prefs)
print("number_of_voters", number_of_voters)
print("number_of_candidates", number_of_candidates)
print("all profiles:", math.factorial(number_of_candidates) ** number_of_voters)
print("normalized:", len(normalized_vectors))
print("roots: ", len(roots))
>>>>>>> c7150ada4e0a828297b6cedea1aba187480e83db

# We need this two to represent the profiles from vectors. generate_roots function creates them internally. 

if print_profiles:
    candidates = get_alphabet_firstn(number_of_candidates)
    possible_prefs = all_possible_prefs(candidates)
    print("\n" * 2)
    print('roots in preference form')
    print('-' * 10)

    for ind, row in enumerate(roots):
        print("Root", ind+1,": ", summarize_voters(comp_vector_to_profile(row, possible_prefs)))



print("\n" * 2)

print('numbers')
print('-' * 10)

print("number of voters: ", number_of_voters)
print("number of candidates: ", number_of_candidates)
# Number of all profiles
print("all profiles:", math.factorial(number_of_candidates) ** number_of_voters)
print("normalized: ", len(normalized))
print("roots: ", len(roots))
