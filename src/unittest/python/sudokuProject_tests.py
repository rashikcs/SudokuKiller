import unittest
import computer_vision
import solve_sudoku
import numpy as np

class SudokuProjectTest(unittest.TestCase):

	snap_path = 'src/main/python/images/image9.jpg'
	extracted_digits_realValue = np.array([0,4,0,6,9,0,5,2,0,0,0,0,3,0,0,6,
									0,1,0,0,0,0,0,0,0,9,0,0,0,0,0,0,3,0,1,9,
									8,7,0,0,0,0,0,6,5,2,9,0,8,0,0,0,0,0,0,6,
									0,0,0,0,0,0,0,9,0,7,0,0,4,0,0,0,0,1,5,0,
									6,9,0,8,0])

	def test_should_issue_computer_vision_board_extraction(self):
		vision = computer_vision.ComputerVision()
		original = vision.read_image(self.snap_path)
		extracted_digits, cropped_board = vision.get_extracted_digits(original, False)
		assert (np.array_equal(extracted_digits, self.extracted_digits_realValue))
		
	def test_should_check_sudoku_solution(self):
		SOLVE_SUDOKU = solve_sudoku.SolveSudoku()
		sudoku_solution = SOLVE_SUDOKU.solve_puzzle(self.extracted_digits_realValue)
		sudoku_solution_real_value = dict({'A1': '1', 'A2': '4', 'A3': '8', 'A4': '6', 'A5': '9', 'A6': '7',\
									  'A7': '5', 'A8': '2', 'A9': '3', 'B1': '7', 'B2': '2', 'B3': '9',\
									  'B4': '3', 'B5': '8', 'B6': '5', 'B7': '6', 'B8': '4', 'B9': '1',\
									  'C1': '5', 'C2': '3', 'C3': '6', 'C4': '4', 'C5': '1', 'C6': '2',\
									  'C7': '7', 'C8': '9', 'C9': '8', 'D1': '6', 'D2': '5', 'D3': '4',\
									  'D4': '2', 'D5': '7', 'D6': '3', 'D7': '8', 'D8': '1', 'D9': '9',\
									  'E1': '8', 'E2': '7', 'E3': '3', 'E4': '9', 'E5': '4', 'E6': '1',\
									  'E7': '2', 'E8': '6', 'E9': '5', 'F1': '2', 'F2': '9', 'F3': '1',\
									  'F4': '8', 'F5': '5', 'F6': '6', 'F7': '3', 'F8': '7', 'F9': '4',\
									  'G1': '4', 'G2': '6', 'G3': '2', 'G4': '1', 'G5': '3', 'G6': '8',\
									  'G7': '9', 'G8': '5', 'G9': '7', 'H1': '9', 'H2': '8', 'H3': '7',\
									  'H4': '5', 'H5': '2', 'H6': '4', 'H7': '1', 'H8': '3', 'H9': '6',\
									  'I1': '3', 'I2': '1', 'I3': '5', 'I4': '7', 'I5': '6', 'I6': '9',\
									  'I7': '4', 'I8': '8', 'I9': '2'})
		assert (sudoku_solution_real_value==sudoku_solution)