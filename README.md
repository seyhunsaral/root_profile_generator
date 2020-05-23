# root_profile_generator

* Requires python3.8 because it uses 'math.comb' function which was introduced there. 
* Requres `sympy` and `numpy`. Dependencies can be installed with `pip install -r requirements.txt`

* roots can be generatated by 
```
roots = generate_roots(number_of_voters, number_of_candidates)
```

* normalized vectors can be generated by one of the three methods

```
normalized = generate_normalized_semidirect(number_of_candidates, number_of_voters)
normalized = generate_normalized_direct(number_of_candidates, number_of_voters)
normalized = generate_normalized_indirect(number_of_candidates, number_of_voters)
```
I will probably create one wrapper to it with the method parameter (as it is in the roots function)

TODO: Changing types to `dtype=np.int16` from `dtype=int` increases the available memory. I think in principle it is possible to use 8 bit as well as long as we have less then 127 voters. 
