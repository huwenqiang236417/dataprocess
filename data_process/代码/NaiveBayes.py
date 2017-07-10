#-*- coding: utf-8 -*-
#建立朴素贝叶斯分类模型
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import time

print ("Scripts starts...")
start = time.time()

inputfile='data.xls' #数据
outputfile = 'bayes.xls' #模型输出文件
data = pd.read_excel(inputfile) #读入数据
#p=0.6
#train = data[:int(len(data)*p),:]
#test = data[int(len(data)*p):,:]
y_train = data.iloc[0:9999,62].as_matrix() #训练样本标签列
x_train = data.iloc[0:9999,0:46].as_matrix() #训练样本特征
y_test = data.iloc[:,62].as_matrix() #测试样本标签列
x_test = data.iloc[:,0:46].as_matrix() #测试样本特征

clf = MultinomialNB().fit(x_train, y_train)  #训练多项式贝叶斯分类器模型 
clf.predict(x_test)  #预测结果
end1 = time.time()
print ("modeltime: %f s" %(end1-start))

count = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x_test), y_test):
      if left == right:
            count += 1
print("预测准确度为：%f" %(float(count)/len(y_test)))

r = pd.DataFrame(clf.predict(x_test), columns = [u'预测结果'])
pd.concat([data.iloc[:,:63], r], axis = 1).to_excel(outputfile)

end2 = time.time()
print ("endtime: %f s" %(end2-start))
print ("Script ends...")