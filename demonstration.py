from Perception import Perceptor
import time

perceptor_1 = Perceptor(10)

perceptor_1.object_to_workspace(demonstration=True)

time.sleep(2)

"""
True values for demonstration object: Cylinder of height 3cm and radius 5cm at position (0.1, 0.1, 0.665)
"""


DEFAULT_SIZE = [0.025, 0.1, 0.002]
DEFAULT_THRESHOLD = 0.01
DEFAULT_CHANGE_THRESHOLD = 0.01


perceptor_1.size_parameters = DEFAULT_SIZE 						# Size parameters [start, end, step]
perceptor_1.error_threshold = DEFAULT_THRESHOLD					# Error threshold at which to stop completely
perceptor_1.error_change_threshold = DEFAULT_CHANGE_THRESHOLD	# Change in error at which to stop current size iteration

perceptor_1.approximate_only()


NEW_SIZE = [0.025, 0.1, 0.002]
NEW_THRESHOLD = 0.01
NEW_CHANGE_THRESHOLD = 0.001


perceptor_1.size_parameters = NEW_SIZE 							# Size parameters [start, end, step]
perceptor_1.error_threshold = NEW_THRESHOLD						# Error threshold at which to stop completely
perceptor_1.error_change_threshold = NEW_CHANGE_THRESHOLD		# Change in error at which to stop current size iteration

perceptor_1.approximate_only()

