# SudokuKiller
Solves sudoku puzzle from image.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine.

```
git clone https://github.com/rashikcs/SudokuKiller.git
```

### Prerequisite Libraries

Libraries used for this project.
```
  - python 3.6
  - pybuilder
  - numpy
  - matplotlib
  - opencv-python==3.4.4.19
  - tensorflow==1.12.0
```

## Running & Building the project

To build the projcet and run the automated tests you neeed to install [Pybuilder](https://pybuilder.io/documentation/installation). PyBuilder is a software build tool written in 100% pure Python, mainly targeting Python applications. PyBuilder is based on the concept of dependency based programming, but it also comes with a powerful plugin mechanism, allowing the construction of build life cycles similar to those known from other famous (Java) build tools.

For a successfull build run following command in the root directory of the project

```
pyb_
```
You should find the 1st version folder in the "target/dist" directory. Start experimenting by running:
```
python start_sudoku_solver.py images/image1.jpg
```
## Acknowledgments

* Inspiration
  https://github.com/mineshpatel1/sudoku-solver