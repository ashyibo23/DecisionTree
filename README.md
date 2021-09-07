# DecisionTree
- :evergreen_tree: implementation of decision tree  from scratch

- Classic Pacman arcade game as developed at UC Berkeley (http://ai.berkeley.edu).

The following functions and classes define methods necessary for building a decision tree classifier, then 
implementing an ensemble of these trees via bootstrap aggregation in order to return a single classification label
as agreed upon by a maximum vote of the ensemble.


Start with Pacman
The Pacman code was developed at UC Berkeley for their AI
course. The folk who developed this code then kindly made it available to everyone. The homepage
for the Berkeley AI Pacman projects is here:
http://ai.berkeley.edu/


From the command line (you will need to use the command line in order to use the various
options), switch to the folder pacman.
Now type:
python pacman.py


Code to control Pacman

python pacman.py --pacman RandomAgent


Towards a classifier 

The file to look for is classifierAgents.py
To run the code in classifierAgents.py, you use:
python pacman.py --pacman ClassifierAgent

