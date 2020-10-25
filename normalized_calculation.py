import rootprofiles as rp
import numpy as np
from collections import Counter
import math

number_of_candidates = 4
number_of_voters = 3

print(len(rp.generate_normalized_direct(number_of_voters,number_of_candidates)),
      rp.num_normalized(number_of_voters,number_of_candidates))


for n in [2,3,4]:
    for m in range(3,26):
        print(n,m, rp.num_normalized(m,n))
