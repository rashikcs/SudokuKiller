#!/usr/bin/env python
# coding: utf-

def cross(str_a, str_b):
	return [a + b for a in str_a for b in str_b]

class SolveSudoku:

	coords = []
	groups = {}
	all_units = []

	def __repr__(self):
		return "An instance of SolveSudoku class."

	def build_sudoku_board(self):
		"""Gets a tuple of reference objects that are useful for describing the Sudoku grid."""

		all_rows = 'ABCDEFGHI'
		all_cols = '123456789'


		coords = cross(all_rows, all_cols)
		row_units = [cross(row, all_cols) for row in all_rows]
		col_units = [cross(all_rows, col) for col in all_cols]


		box_units = [cross(row_square, col_square) for row_square in ['ABC', 'DEF', 'GHI']
															for col_square in ['123', '456', '789']]


		all_units = row_units + col_units + box_units  # Add units together
		groups = {}

		groups['units'] = {pos: [unit for unit in all_units if pos in unit] for pos in coords}

		groups['peers'] = {pos: set(sum(groups['units'][pos], [])) - {pos} for pos in coords}
		return coords, groups, all_units

	def __init__(self):
		self.coords, self.groups, self.all_units = self.build_sudoku_board()

	def construct_grid_row(self, col_counter, grid, row, col, display, width):
		null_chars = '0.'
		col_counter += 1
		if grid[row + col] in null_chars:
			grid[row + col] = '.'

		display += ('%s' % grid[row + col]).center(width)
		if col_counter % 3 == 0 and col_counter % 9 != 0:
			display += '|'
		if col_counter % 9 == 0:
			display += '\n'
		return display, col_counter

	def construct_complete_grid(self, display, have_coords, grid, width):

		all_rows = 'ABCDEFGHI'
		all_cols = '123456789'
		row_counter = 0
		col_counter = 0

		for row in all_rows:
			if have_coords:
				display += all_rows[row_counter] + ' |'
			row_counter += 1
			for col in all_cols:
				display, col_counter = self.construct_grid_row(col_counter, grid, row, col, display, width)

			if row_counter % 3 == 0 and row_counter != 9:
				if have_coords:
					display += '  |'
				display += '+'.join([''.join(['-' for x in range(width * 3)])\
									for y in range(3)]) + '\n'
		return display

	def display_grid(self, grid, have_coords=False):
		"""
		Displays a 9x9 soduku grid in a nicely formatted way.
		"""
		if grid is None or grid is False:
			return None

		all_cols = '123456789'

		if type(grid) == str:
			grid = self.parse_puzzle(grid)
		elif type(grid) == list:
			grid = self.parse_puzzle(''.join([str(el) for el in grid]))

		width = max([3, max([len(grid[pos]) for pos in grid]) + 1])
		display = ''

		if have_coords:
			display += '   ' + ''.join([all_cols[i].center(width) for i in range(3)]) + '|'
			display += ''.join([all_cols[i].center(width) for i in range(3, 6)]) + '|'
			display += ''.join([all_cols[i].center(width) for i in range(6, 9)]) + '\n   '
			display += '--' + ''.join(['-' for x in range(width * 9)]) + '\n'

		display = self.construct_complete_grid(display, have_coords, grid, width)

		print(display)
		return display

	def parse_puzzle(self, puzzle, digits='123456789', nulls='0'):
		'''
		Parses a string describing a Sudoku puzzle board
		into a dictionary with each cell mapped to its relevant
		coordinate, i.e. A1, A2, A3...

		# Serialise the input into a string, let
		the position define the grid location and .0 can be empty positions
		# Ignore any characters that aren't digit input or nulls'''

		flat_puzzle = ['.' if str(char) in nulls else str(char) for char in puzzle if str(char) in digits + nulls]

		if len(flat_puzzle) != 81:
			raise ValueError('Input puzzle has %s grid positions specified, must be 81.'
							' Specify a position using any digit from 1-9 and 0 or . '
							'for empty positions.' % len(flat_puzzle))

		return dict(zip(self.coords, flat_puzzle))

	def validate_sudoku(self, puzzle):
		"""Checks if a completed Sudoku puzzle has a valid solution."""
		if puzzle is None:
			return False

		full = [str(x) for x in range(1, 10)]  # Full set, 1-9 as strings

		return all([sorted([puzzle[cell] for cell in unit]) == full for unit in self.all_units])

	def eliminate(self, grid, pos, val):
		"""Eliminates `val` as a possibility from all peers of `pos`."""
		if grid is None: 
			return None

		if val not in grid[pos]:
			return grid
		grid[pos] = grid[pos].replace(val, '')
		if len(grid[pos]) == 0:
			return None
		elif len(grid[pos]) == 1:
			for peer in self.groups['peers'][pos]:
				grid = self.eliminate(grid, peer, grid[pos])# Recurses, propagating the constraint
				if grid is None:
					return None
		for unit in self.groups['units'][pos]:
			possibilities = [p for p in unit if val in grid[p]]
			if len(possibilities) == 0 or (len(possibilities) == 1 and len(grid[possibilities[0]]) > 1 and \
			self.confirm_value(grid, possibilities[0], val) is None):
				return None
		return grid


	def confirm_value(self, grid, pos, val):
		"""Confirms a value by eliminating all other remaining possibilities."""
		remaining_values = grid[pos].replace(val, '')  # Possibilities we can eliminate due to the confirmation
		for val in remaining_values:
			grid = self.eliminate(grid, pos, val)
		return grid


	def solve_puzzle(self, puzzle):
		"""Solves a Sudoku puzzle from a string input."""
		digits = '123456789' 

		input_grid = self.parse_puzzle(puzzle)
		input_grid = {k: v for k, v in input_grid.items() if v != '.'}  # Filter so we only have confirmed cells
		output_grid = {cell: digits for cell in self.coords}  # Create a board where all digits are possible in each cell


		for position, value in input_grid.items():
			output_grid = self.confirm_value(output_grid, position, value)

		if self.validate_sudoku(output_grid):
			return output_grid

		def guess_digit(grid):
			"""Guesses a digit from the cell with the fewest unconfirmed possibilities and propagates the constraints."""

			if grid is None:
				return None

			# Reached a valid solution, can end
			if all([len(possibilities) == 1 for cell, possibilities in grid.items()]):
				return grid

			_, pos = min([(len(possibilities), cell) for cell, possibilities in grid.items() if len(possibilities) > 1])

			for val in grid[pos]:
				solution = guess_digit(confirm_value(grid.copy(), pos, val))
				if solution is not None:
					return solution

		output_grid = guess_digit(output_grid)
		return output_grid