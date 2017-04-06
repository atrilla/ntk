# NTK - Neural Network Toolkit

The Neural Network Toolkit (NTK) project develops open-source software
for neural computation. Experimental code developed in Python. Performance
code developed in C.

## Artificial Neural Networks

Artificial Neural Networks are powerful models in Artificial Intelligence 
and Machine Learning because they are suitable for many scenarios. Their 
elementary working form is direct and simple. However, the devil is in the 
details, and these models are particularly in need of much empirical 
expertise to get tuned adequately so as to succeed in solving the problems 
at hand. 

The neural network is plausibly renown to be the universal learning system. 
Without loss of generality, this project makes some decisions regarding 
the model shape/topology, the training method, and the like. These design 
choices, though, are easily tweaked so that the same implementation may be 
suitable to solve all kinds of problems. This is accomplished by first 
breaking down its complexity, and then by depicting a procedure to tackle 
problems systematically in order to quickly detect model flaws and fix them 
as soon as possible. Let’s say that the gist of this process is to achieve 
a “lean adaptation” procedure for neural networks.


## Theory of Operation

Artificial Neural Networks (ANNs) are interesting models in Artificial 
Intelligence and Machine Learning because they are powerful enough to succeed 
at solving many different problems. Historical evidence of their importance 
can be found as most leading technical books dedicate many pages to cover 
them comprehensibly.

Overall, ANNs are general-purpose universal learners driven by data. They 
conform to the connectionist learning approach, which is based on an 
interconnected network of simple units. 
![Multilayer Perceptron, one of the most widely applied neural network models.](https://github.com/atrilla/ntk/blob/master/images/mlp_framework.png)
Such simple units, aka neurons, 
compute a nonlinear function over the weighted sum of their inputs. Neural 
networks are expressive enough to fit to any dataset at hand, and yet they 
are flexible enough to generalise their performance to new unseen data. It 
is true, though, that neural networks are fraught with experimental details 
and experience makes the difference between a successful model and a skewed 
one.

## Notebooks

* [Multilayer Perceptron](https://github.com/atrilla/ntk/blob/master/explore/Multilayer.ipynb)

## Requirements

Python:

* Python 2.7.6
* Numpy 1.8.2
* Scikit-learn 2.7.6


C:

* gcc 4.8.4
* GNU Make 3.81

