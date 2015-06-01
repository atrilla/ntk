NTK - Neural Network Toolkit
============================

The Neural Network Toolkit (NTK) project develops open-source software
for neural computation.

The NTK software toolkit is a framework that facilitates the creation 
and evaluation of bespoke neural network models. It is designed to be 
flexible and adaptable. Developed in C for intensive computations.


Mission
-------
To provide a highly performant open-source implementation of neural
networks. Related references:

* Hertz, J., Krogh, A. and Palmer, R. G., "Introduction to the theory
of neural computation", Addison-Wesley, 1991.


Features
--------
All algorithms described in the aforementioned references (planned).


Developer guidelines
--------------------
### Install ###
Requirements:

* gcc 4.6.3
* GNU Make 3.81
* Doxygen 1.7.6.1

Compilation:

* Binaries (library and tests): make
* Documentation: make doc
* Cleaning: make clean

### Hacking ###
File organisation of this source code distribution of NTK:

* **ntk/** Root directory of the NTK project.
* **build/** Generated object code binaries.
* **ChangeLog** Record of source changes.
* **config/** Configuration files.
* **COPYING** Rights granted with this source code distribution.
* **dist/** Built library.
* **doc/** Generated documentation in HTML format.
* **include/** Source headers.
* **Makefile** Build process instructions.
* **README.md** General description of NTK.
* **src/** Source code files.
* **test/** Sources of the tests.


Development status
------------------
* NTK v1.0, under development.


Glossary
--------
* **McCulloch-Pitts unit** Synchronous discrete-time binary threshold
unit (step function, or Heaviside function).

* **Real neuron** The unit fires a pulse or action potential of fixed
strength and duration if the weighted sum of the inputs reaches or
exceeds a threshold. After firing, the unit has to wait for the
refractory period before it can fire again. This can also be
interpreted as a continuous graded response with hysteresis. It thus
produces a sequence of pulses (magnitude and phase). It has some
response delay (asynchronous update).

* **Unit** General computation element. Its state or activation is
given by a nonlinear function (activation func, gain func, transfer
func or squashing func). Updated asynchronously.

* **Updating** Computing unit response. It can be synchronous (all
units have the same delay) or asynchronous (random order at random
times).

* **Weight** Represents the strength of the connection between two
units. It can the positive or negative.


Contact
-------
FAQ, website and mailing list may be created if many users/developers
join the project.

In the meantime, for any comment or suggestion of any kind, please
contact Alexandre Trilla <alex@atrilla.net>.

