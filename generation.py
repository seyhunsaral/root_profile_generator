import rootprofiles as rp

# End of functions
#===============================================================================
#=====================  Generation                   ===========================
#===============================================================================

number_of_candidates = 3
number_of_voters = 3
print_profiles =  False # or it will just show the numbers 

# Generate Normalized (for demonstration only, it is created by generate_roots function)
#normalized = generate_normalized_semidirect(number_of_candidates, number_of_voters)

# Roots
# If you are running calculations, it is better comment out normalized genration, or to generate roots from normalized vector direclty by "get_root_from_normalized_vector" function in order not to calculate the same thing twise.
roots = rp.generate_roots(number_of_voters, number_of_candidates, method="direct")

if print_profiles:
    print("\n" * 2)
    print('roots in vector form')
    print('-' * 10)
    print(roots)

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
#print("normalized: ", len(normalized))
print("roots: ", len(roots))
