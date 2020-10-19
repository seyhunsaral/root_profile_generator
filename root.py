import argparse
import rootprofiles as rp
import numpy as np
import math
from pprint import pprint

def fullprint(*args, **kwargs):
#https://stackoverflow.com/a/24542498/1819625
  opt = np.get_printoptions()
  np.set_printoptions(threshold=np.inf)
  pprint(*args, **kwargs)
  np.set_printoptions(**opt)

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--voters" , type=int)
parser.add_argument("-c", "--candidates" , type=int)
parser.add_argument("-s", "--show", action="store_true")
parser.add_argument("-w", "--write", action="store_true")

args = parser.parse_args()

number_of_candidates = args.candidates
number_of_voters = args.voters
show = bool(args.show)

print_profiles =  False # or it will just show the numbers 

print("Starting", number_of_voters, number_of_candidates)
roots = rp.generate_roots(number_of_voters, number_of_candidates, method="direct")

print("Completed")


print("All  :", math.factorial(number_of_candidates) ** number_of_voters)
print("Roots: " + str(len(roots)))

if args.show:
    # There might be a better version of it
    sorted_roots = roots[np.argsort(roots[:, 0])]
    fullprint(sorted_roots)
    print("\n")

if args.write:
    filename = "roots_" + str(number_of_candidates) + "_" + str(number_of_voters) + ".csv"
    print("writing to:", filename)
    np.savetxt(filename, roots, fmt='%i', delimiter=",")

