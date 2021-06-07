# SudokuKiller
Solves sudoku puzzle from image.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine.

```
git clone https://github.com/rashikcs/SudokuKiller.git
```

### Prerequisite Libraries

Libraries used and installed for this project.
```
  - python 3.6
  - pybuilder == 0.11.17
  - numpy
  - matplotlib
  - opencv-python==3.4.4.19
  - tensorflow==1.12.0
```
One just need to have python and mentioned pybuilder to proceed further and the rest will be installed autamatically. 
## Running & Building the project

To build the projcet and run the unit tests you neeed to install [Pybuilder== 0.11.17](https://pybuilder.io/documentation/installation) using pip or conda. PyBuilder is a software build tool written in 100% pure Python, mainly targeting Python applications. PyBuilder is based on the concept of dependency based programming, but it also comes with a powerful plugin mechanism, allowing the construction of build life cycles similar to those known from other famous (Java) build tools.

For a successfull build run following command in the root directory of the project

For windows
```
pyb_
```
For Linux
```
pyb
```
You should find the 1st version of the release in the "target/dist/sudokukiller-1.0.dev0/" directory. Start experimenting by running the following script with provided image:
```
python start_sudoku_solver.py images/image1.jpg
```

## Continuous Integration: Travis CI
A hosted, distributed continuous integration service is used to build and test software
projects hosted at GitHub. It will build an app and run all tests each time after a successful push
to the repository and generate report. It was used to in this project to ensure continuous integration of the project

## Acknowledgments

* Inspiration
  https://github.com/mineshpatel1/sudoku-solver