from CSIResNet import ResNet,BasicBlock,BottleNeck
from LoadMyData import MyData
import paddle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from visualdl import LogWriter
import numpy as np
import itertools
import os
from itertools import chain 
import pandas as pd
root_dir='data/data140457/CSI-GAF4/data'
test_dir='target_domain'
# visualdl service upload --logdir
batsize=50

testdata=MyData(root_dir,test_dir,4)
test_loader = paddle.io.DataLoader(testdata, batch_size=batsize, shuffle=False,drop_last=True)

#测试
y_prel=[]
y_result=[]
model=ResNet(BasicBlock,[1,1,1,1],6)
model_dict = paddle.load('model1.pdparams')
model.set_state_dict(model_dict)
correct_t = 0
model.eval()
test_acc=[]
use_gpu = True
paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
for batch_id, data in enumerate(test_loader()):
        img = data[0]
        label = data[1] 
        # 计算模型输出
        predict_label = model(img)
        correct_t=paddle.metric.accuracy(predict_label,label)
        predict_label=np.argmax(predict_label,axis=1)
        y_prel.append(predict_label)
        
        label=label.numpy()
        label=label.flatten()
        # print(label)
        y_result.append(label)
        
        test_acc.append(correct_t.numpy())

y_prel=np.array(y_prel)
y_prel=y_prel.flatten()
y_result=np.array(y_result)
y_result=y_result.flatten()

    

print(np.mean(test_acc))

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
labels1=['Push&Pull', 'Sweep', 'Clap', 'Slide', 'Draw-Z', 'Draw-N']
tick_marks = np.array(range(len(labels1))) + 0.5
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels1)))
    plt.xticks(xlocations, labels1, rotation=0,size=14)
    plt.yticks(xlocations, labels1,size=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm = confusion_matrix(y_result, y_prel)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 8), dpi=120)
ind_array = np.arange(len(labels1))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
plot_confusion_matrix(cm_normalized, title='confusion matrix')
# show confusion matrix
plt.savefig('./result/confusion_matrix2.png', format='png')
plt.show()
y_prel=pd.DataFrame(y_prel) 
y_result=pd.DataFrame(y_result)
y_prel.to_csv('./result/prel0.csv')
y_result.to_csv('./result/result0.csv')