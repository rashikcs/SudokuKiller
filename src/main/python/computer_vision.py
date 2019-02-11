import operator
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

from neural_net.DigitRecogniser import DigitRecogniser


class ComputerVision:

	def __repr__(self):
		return "Module of image processing tasks."

	def read_image(self, path):
		return cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

	def display_rects(self, in_img, rects, colour=255):
		img = in_img.copy()
		for rect in rects:
			img = cv2.rectangle(img, tuple(int(x) for x in rect[0]),\
			tuple(int(x) for x in rect[1]), colour)
		self.show_image(img, "With squares")
		return img

	def infer_grid(self, img):
		'''Infers 81 cell grid from a square image.'''
		squares = []
		side = img.shape[:1]
		side = side[0] / 9
		for i in range(9):
			for j in range(9):
				top_corner = (j * side, i * side)#Top left corner of a bounding box
				bottom_corrner = ((j + 1) * side, (i + 1) * side) # Bottom right corner of bounding box
				squares.append((top_corner, bottom_corrner))
		return squares

	def distance_between(self, p_1, p_2):
		dis_1 = p_2[0] - p_1[0]
		dis_2 = p_2[1] - p_1[1]
		return np.sqrt((dis_1 ** 2) + (dis_2 ** 2))


	def crop_and_warp(self, img, crop_rect):
		'''Crops and warps a rectangular section from an image into a square of similar size.'''

		top_left, top_right, bottom_right, bottom_left = crop_rect[0],\
crop_rect[1], crop_rect[2], crop_rect[3]

		src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

		side = max([
			self.distance_between(bottom_right, top_right),
			self.distance_between(top_left, bottom_left),
			self.distance_between(bottom_right, bottom_left),
			self.distance_between(top_left, top_right)
		])

		dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

		perspective_transformed = cv2.getPerspectiveTransform(src, dst)

		return cv2.warpPerspective(img, perspective_transformed, (int(side), int(side)))

	def find_corners_of_largest_polygon(self, img):
		'''Finds the 4 extreme corners of the largest contour in the image.'''
		_, contours, h = cv2.findContours(img.copy(),\
		cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
		contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
		polygon = contours[0]  # Largest image

		bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon])\
		, key=operator.itemgetter(1))
		top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]),\
		key=operator.itemgetter(1))
		bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]),\
		key=operator.itemgetter(1))
		top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]),\
		key=operator.itemgetter(1))


		return [polygon[top_left][0], polygon[top_right][0],\
		polygon[bottom_right][0], polygon[bottom_left][0]]

	def display_points(self, in_img, points, radius=5, colour=(0, 0, 255)):
		'''Draws circular points on an image.'''
		img = in_img.copy()

		if len(colour) == 3 and len(img.shape) == 2 or img.shape[2] == 1:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

		for point in points:
			img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
		self.show_image(img, "With points")
		return img

	def show_image(self, img, title):
		cv2.namedWindow(title, cv2.WINDOW_NORMAL)
		cv2.imshow(title, img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def show_digits(self, digits, colour=255):
		rows = []
		with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1,\
		cv2.BORDER_CONSTANT, None, colour) for img in digits]
		for i in range(9):
			row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
			rows.append(row)
		self.show_image(np.concatenate(rows), "Extracted Digits")

	def cut_from_rect(self, img, rect):
		return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


	def scale_and_centre(self, img, size, margin=0, background=0):
		'''Scales and centres an image onto a new background square.'''
		height, width = img.shape[:2]

		def centre_pad(length):
			'''Handles centering for a given length that may be odd or even.'''
			if length % 2 == 0:
				side1 = int((size - length) / 2)
				side2 = side1
			else:
				side1 = int((size - length) / 2)
				side2 = side1 + 1
			return side1, side2

		def scale(row, x_value):
			return int(row * x_value)

		if height > width:
			t_pad = int(margin / 2)
			b_pad = t_pad
			ratio = (size - margin) / height
			width, height = scale(ratio, width), scale(ratio, height)
			l_pad, r_pad = centre_pad(width)
		else:
			l_pad = int(margin / 2)
			r_pad = l_pad
			ratio = (size - margin) / width
			width, height = scale(ratio, width), scale(ratio, height)
			t_pad, b_pad = centre_pad(height)

		img = cv2.resize(img, (width, height))
		img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
		return cv2.resize(img, (size, size))

	# Gets the maximum bound area which should be the grid of the image
	def get_max_area(self, img, scan_br, scan_tl, ):
		for x in range(scan_tl[0], scan_br[0]):
			for y in range(scan_tl[1], scan_br[1]):
				if img.item(y, x) == 255 and x < width and y < height:
					area = cv2.floodFill(img, None, (x, y), 64)
					if area[0] > max_area:
						max_area = area[0]
						return (x, y)

	def get_main_feature_in_grayscale(self, img, height, width, seed_point):
		
		self.dilate_featureless_area(img, height, width)
		mask = np.zeros((height + 2, width + 2), np.uint8)

		if all([p is not None for p in seed_point]):
			cv2.floodFill(img, mask, seed_point, 255)
		return mask

	def hide_featureless_area(self, img, x, y, mask):
		if img.item(y, x) == 64:  # Hide anything that isn't the main feature
			cv2.floodFill(img, mask, (x, y), 0)

	def get_bbox_of_feature(self, img, x, y, area):
		# Find the bounding parameters
		if img.item(y, x) == 255:
			top = y if y < area[0] else area[0]
			bottom = y if y > area[1] else area[1]
			left = x if x < area[2] else area[2]
			right = x if x > area[3] else area[3]
		return top, left, right, bottom

	def find_largest_feature(self, inp_img, scan_tl=None, scan_br=None):

		'''
		Uses the fact the `floodFill` function returns a
		bounding box of the area it filled to find the biggest
		connected pixel structure in the image.
		Fills this structure in white, reducing the rest to black.
		'''

		img = inp_img.copy()  # Copy the image, leaving the original untouched
		height, width = img.shape[:2]

		max_area = 0
		seed_point = (None, None)

		if scan_tl is None:
			scan_tl = [0, 0]

		if scan_br is None:
			scan_br = [width, height]

		seed_point = self.get_max_area(img, scan_br, scan_tl, max_area)

		mask = self.get_main_feature_in_grayscale(img, height, width, seed_point)

		top, bottom, left, right = height, 0, width, 0

		for x in range(width):
			for y in range(height):
				self.hide_featureless_area(img, x, y, mask)
				top, left, right, bottom = self.get_bbox_of_feature(img, x, y, [top, bottom, left, right] )

		bbox = [[left, top], [right, bottom]]

		return img, np.array(bbox, dtype='float32'), seed_point

	def dilate_featureless_area(self, img, height, width):
		# Colour everything grey
		for x in range(width):
			for y in range(height):
				if img.item(y, x) == 255 and x < width and y < height:
					cv2.floodFill(img, None, (x, y), 64)

	def get_highlighted_measurements(self, x, y, measurements):
		if y < measurements[0]:
			measurements[0] = y

		if y > measurements[3]:
			measurements[3] = y

		if x < measurements[1]:
			measurements[1] = x

		if x > measurements[2]:
			measurements[2] = x
		return measurements

	def get_highlighted_points(self, x, y, points):
		if x + y < sum(points[0]):
			points[0] = [x, y]

		if x + y > sum(points[2]):
			points[2] = [x, y]

		if x - y > points[1][0] - points[1][1]:
			points[1] = [x, y]

		if x - y < points[3][0] - points[3][1]:
					points[3] = [x, y]
		return points

	def get_highlighted_bounding_positions(self, img, measurements, points, bbox):
		# If it is a highlighted point, use it to determine the bounding position
		if img.item(y, x) == 255:
			if bbox:
				return self.get_highlighted_measurements(x, y, measurements), points
			else:
				return measurements, self.get_highlighted_points(x, y, points)

	def get_bbox_from_seed(self, inp_img, seed, bbox=True):
		img = inp_img.copy() 
		height, width = img.shape[:2]
		mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image
		
		self.dilate_featureless_area(img, height, width)
		# Highlight the main feature
		if all([p is not None for p in seed]):
			cv2.floodFill(img, mask, seed, 255)

		top_left = [width, height]
		top_right = [0, height]
		bottom_left = [width, 0]
		bottom_right = [0, 0]

		top = height
		bottom = 0
		left = width
		right = 0

		for x in range(width):
			for y in range(height):
				if img.item(y, x) == 64:  # Hide anything that isn't the main feature
					cv2.floodFill(img, mask, (x, y), 0)

				measurements, points = self.get_highlighted_bounding_positions(img, [top, left, right, bottom],[top_left, top_right, bottom_right, bottom_left], bbox)

		top_left = measurements[0]
		top_right = measurements[1]
		bottom_right = measurements[2]
		bottom_left = measurements[3]

		top = measurements[0]
		left = measurements[1]
		right = measurements[2]
		bottom = measurements[3]

		if bbox:
			top_left = [left, top]
			bottom_right = [right, bottom]
			rect = [top_left, bottom_right]  # Only need the top left and bottom right points
		else:
			rect = [top_left, top_right, bottom_right, bottom_left]

		return img, np.array(rect, dtype='float32')

	def create_blank_image(self, width, height, grayscale=True, include_gray_channel=False):
		if grayscale:
			if include_gray_channel:
				return np.zeros((height, width, 1), np.uint8)
			else:
				return np.zeros((height, width), np.uint8)
		else:
			return np.zeros((height, width, 3), np.uint8)
		
	def extract_cell_raw(self, img, rect, size, include_gray_channel=False):
		cell = self.cut_from_rect(img, rect)
		cell = cv2.resize(cell, (size, size))
		if include_gray_channel:
			cell = cell.reshape((size, size, 1))
		#self.show_image(cell,"cropped cell")
		return cell
		
	def extract_digit(self, img, rect, size, mode, include_gray_channel=False):

		def grid_square_threshold(digit, mode):
			"""Thresholding algorithm for a single digit or grid square."""
			if 'basic' in mode:
				return digit

			if 'blur' in mode:
				digit = cv2.GaussianBlur(digit, (3, 3), 0)

			if 'otsu' in mode:
				ret, digit = cv2.threshold(digit, 5, 255, cv2.THRESH_OTSU)

			if 'adaptive' in mode:
				digit = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
			digit = cv2.bitwise_not(digit, digit)
			return digit

		
		digit = self.cut_from_rect(img, rect)  # Get the digit box from the whole square


		digit = grid_square_threshold(digit, mode)

		if 'cell' in mode:
			digit = cv2.resize(digit, (28, 28))
			return digit

		# Use fill feature finding to get the largest feature in middle of the box
		# Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
		h, w = digit.shape[:2]
		margin = int(np.mean([h, w]) / 2.5)
		discard, bbox, seed = self.find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
		digit, bbox = self.get_bbox_from_seed(digit, seed)

		# Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
		w = bbox[1][0] - bbox[0][0]
		h = bbox[1][1] - bbox[0][1]

		# Ignore any small bounding boxes
		if w > 0 and h > 0 and (w * h) > 100:
			digit = self.cut_from_rect(digit, bbox)
			digit = self.scale_and_centre(digit, size, 4)
			if include_gray_channel:
				digit = digit.reshape((size, size, 1))
			return digit
		else:
			return None

	def get_digits(self, img, squares, size, display_image, show=False, include_blanks=False, raw=False):

		if display_image: self.show_image(img,"cropped")
		blank = self.create_blank_image(size, size, include_gray_channel = False)

		digits = []

		for i, rect in enumerate(squares):
			digit = self.extract_cell_raw(img, rect, size, include_gray_channel=False)

			if digit is not None:
				digits.append(digit)
			
			elif include_blanks:
				digits.append(blank)
			#self.show_image(digit, "Digits")
		return np.array(digits)
		

	def pre_process_image(self, img, skip_dilate=False):

		proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

		proc = cv2.adaptiveThreshold(proc, 255,\
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

		proc = cv2.bitwise_not(proc, proc)
		if not skip_dilate:
			kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
			kernel = kernel.astype('uint8')
			proc = cv2.dilate(proc, kernel)
		return proc

	def get_extracted_digits(self, original, show_image=True):
		processed = self.pre_process_image(original)
		corners = self.find_corners_of_largest_polygon(processed)

		cropped_board = self.crop_and_warp(original, corners)
		squares = self.infer_grid(cropped_board)
		#self.show_image(cropped_board,"cropped_board")	
		digits = self.get_digits(cropped_board, squares, 28, show_image)

		model_path = os.path.join(os.path.dirname(__file__), 'best-model', 'model.ckpt')
		digit_recogniser = DigitRecogniser(model_path)
		board_int = [0] * 81
		board_int = digit_recogniser.predict_digit(digits)
		#print("Extracted Digits", board_int, "\n\n")

		if show_image: self.show_digits(digits)
		
		return board_int, cropped_board

	def draw_numbers(self, cropped_board, sudoku_solution, extracted_digits, colour=(255, 0, 0)\
	, thickness=3):
	
		numbers = [v for k, v in sudoku_solution.items()]
		img = cropped_board.copy()
		squares = self.infer_grid(cropped_board)
		
		scale = int((squares[0][1][0] - squares[0][0][0]) * 0.075)  # Dynamic scale for font
		for i, square in enumerate(squares):
			condition = extracted_digits[i] == 0  # Don't draw numbers given on the board

			if condition:
				# Get font height and width
				fh, fw = cv2.getTextSize(str(numbers[i]), cv2.FONT_HERSHEY_PLAIN, scale, thickness)[0]
				h_pad = int((square[1][0] - square[0][0] - fw) / 2)  # Padding to centre the number
				v_pad = int((square[1][1] - square[0][1] - fw) / 2)
				
				img = cv2.putText(img, str(numbers[i]), (int(square[0][0]) + h_pad, int(square[1][1]) - v_pad),
								  cv2.FONT_HERSHEY_PLAIN, fontScale=scale, color=colour, thickness=thickness)
				#self.show_image(img, "draw_numbers")
		return img