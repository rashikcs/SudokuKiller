import sys
import os
import sudoku_snap

def main():
	path = os.path.join('images','image9.jpg')
	if len(sys.argv) > 1 and sys.argv[1] is not None: path = sys.argv[1]
	example = sudoku_snap.SudokuSnap(path)
	example.get_solution()

if __name__ == "__main__":
	main()
