import solve_sudoku
import computer_vision

class SudokuSnap:

	def __init__(self, path):
		if path is not None:
			self.snap_path = path

	def get_solution(self):
		vision = computer_vision.ComputerVision()
		original = vision.read_image(self.snap_path)
		vision.show_image(original, "original")

		extracted_digits, cropped_board = vision.get_extracted_digits(original)

		SOLVE_SUDOKU = solve_sudoku.SolveSudoku()
		sudoku_solution = SOLVE_SUDOKU.solve_puzzle(extracted_digits)
		SOLVE_SUDOKU.display_grid(sudoku_solution, True)

		board_with_answer = vision.draw_numbers(cropped_board, sudoku_solution, extracted_digits)
		vision.show_image(board_with_answer, "Drawn_Numbers")
