# axs
A PyTorch Module Example

	俺寻思这玩意应该能用，实际测试确实有效果。
	main.py脚本中的模型采用一个axs层加一个不进行求导的全连接层，可实现MNIST测试集89%的识别成功率。
	俺寻思通过在模型结构、初始化、暂退（dropout）、训练时调餐速度比率等方面的调整，这玩意应该能够有更广泛的用途。