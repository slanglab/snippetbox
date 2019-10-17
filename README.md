### Other details 

`demo.py` implements greedy algorithm here: https://ryanmcd.github.io/papers/globsumm.pdf

As of Oct 17th
- The salience function just averages the PMI of the top 5 most relevant tokens. 
- The (pairwise) redundancy function is just the Jaccard index of the tokens in two sentences

### Installation

1. Clone, create a python3 virtual enviroment and $pip install -r requirements.txt
2. $ pip install https://github.com/kpu/kenlm/archive/master.zip b/c this package is not on pip 

### Repo

- To see how all runs see `demo.py`
- For KenLM: `klm`
- Preprocessing and experiments: `scripts`
- For snippet algorithms/methods: `code/wellformedness.py`
- Environment variables are in `environments`
