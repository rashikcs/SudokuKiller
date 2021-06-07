from pybuilder.core import init, use_plugin

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.install_dependencies")
use_plugin("python.distutils")

default_task = "publish"

@init
def initialize(project):
	project.build_depends_on('numpy')
	project.depends_on('matplotlib')
	project.depends_on('opencv-python', "==3.4.4.19")
	project.depends_on('tensorflow', "==1.12.0")
	# File is installed relative to sys.prefix
	project.install_file("Lib/site-packages/neural_net","neural_net/DigitRecogniser.py")
	project.install_file("Lib/site-packages/neural_net","neural_net/NeuralNetwork.py")
	project.install_file("Lib/site-packages/best-model","best-model/model.ckpt.data-00000-of-00001")
	project.install_file("Lib/site-packages/best-model","best-model/model.ckpt.index")
	project.install_file("Lib/site-packages/best-model","best-model/model.ckpt.meta")