# quanti-gin

<img src=./LOGO.png width=200px alt="quanti-gin Logo">

A customizable data generator for quantum simulation.
Using this library, you can easily create larger sets of simulations and output data for optimizing molecule ground state.
Per default, we are using [Tequila](https://github.com/tequilahub/tequila) as a quantum backend.

## Installation

To install quanti-gin run `pip install quanti-gin`.
This should install all required packages with quanti-gin.

## Usage

For a basic data generation job use:

    python -m quanti_gin 4 100

The basic format of the command line is:

    python -m quanti_gin <number_of_atoms> <number_of_jobs>

If you want to learn about more parameters you can use help:

    python -m quanti_gin -h

## Customize the data generator

quanti-gin is designed so it can be easily customized with your heuristics for data generation.
You can create your own version by simply subclassing the `quanti_gin.DataGenerator` class.

A full example of this can be found in [examples/customized_generator.py](quanti_gin/examples/customized_generator.py).

You can run the example code by executing:

    python -m quanti_gin.examples.customized_generator
