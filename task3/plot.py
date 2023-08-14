from matplotlib import pyplot as plt

x_label = [0,1,2,3,4,5,6,7,8,9]
train_loss = [0.685, 0.58, 0.364, 0.291, 0.251, 0.211, 0.192, 0.169, 0.159, 0.140]
train_acc = [0.54, 0.69, 0.846, 0.883, 0.902, 0.920, 0.928, 0.937, 0.942, 0.950]
test_loss = [0.71, 0.38, 0.321, 0.313, 0.34, 0.339, 0.267, 0.271, 0.265, 0.273]
test_acc = [0.503, 0.84, 0.860, 0.880, 0.870, 0.857, 0.896, 0.895, 0.894, 0.895]

train_lossG = [0.715, 0.602, 0.452, 0.351, 0.326, 0.265, 0.244, 0.226, 0.211, 0.198]
train_accG = [0.56, 0.70, 0.834, 0.867, 0.886, 0.915, 0.920, 0.928, 0.936, 0.940]
test_lossG = [0.80, 0.41, 0.42, 0.38, 0.35, 0.36, 0.345, 0.312, 0.323, 0.304]
test_accG = [0.512, 0.802, 0.835, 0.825, 0.829, 0.851, 0.872, 0.865, 0.859, 0.870]

plt.plot(x_label, train_lossG, c='blue', marker='o', linestyle=':', label='train_loss')
plt.plot(x_label, test_lossG, c='red', marker='*', linestyle='-', label='test_loss')
plt.legend(loc='upper right')
plt.savefig(fname="GRUloss.png")
#plt.plot(lambda1, params, c='green', marker='+', linestyle='--', label='parameters')
