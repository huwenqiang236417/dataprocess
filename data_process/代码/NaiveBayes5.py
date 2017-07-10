from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import time

print ("Scripts starts...")
start = time.time()

inputfile='./data/test.xls' #数据
data = pd.read_excel(inputfile) #读入数据

y_train1 = data.iloc[0:9999,36].as_matrix() #训练样本标签列
x_train1 = data.iloc[0:9999,0:4].as_matrix() #训练样本特征
y_test1 = data.iloc[:,36].as_matrix() #测试样本标签列
x_test1 = data.iloc[:,0:4].as_matrix() #测试样本特征
clf = MultinomialNB().fit(x_train1,y_train1)
clf.predict(x_test1)
end1 = time.time()
print ("modeltime: %f s" %(end1-start))
count1 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x_test1), y_test1):
      if left == right:
            count1 += 1
print("饮食预测准确度为：%f" %(count1/len(y_test1)))

y_train2 = data.iloc[0:9999,36].as_matrix() #训练样本标签列
x_train2 = data.iloc[0:9999,5:8].as_matrix() #训练样本特征
y_test2 = data.iloc[:,36].as_matrix() #测试样本标签列
x_test2 = data.iloc[:,5:8].as_matrix() #测试样本特征
clf = MultinomialNB().fit(x_train2,y_train2)
clf.predict(x_test2)
end2 = time.time()
print ("modeltime: %f s" %(end2-end1))
count2 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x_test2), y_test2):
      if left == right:
            count2 += 1
print("烟酒预测准确度为：%f" %(count2/len(y_test2)))

y_train3 = data.iloc[0:9999,36].as_matrix() #训练样本标签列
x_train3 = data.iloc[0:9999,9:11].as_matrix() #训练样本特征
y_test3 = data.iloc[:,36].as_matrix() #测试样本标签列
x_test3 = data.iloc[:,9:11].as_matrix() #测试样本特征
clf = MultinomialNB().fit(x_train3,y_train3)
clf.predict(x_test3)
end3 = time.time()
print ("modeltime: %f s" %(end3-end2))
count3 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x_test3), y_test3):
      if left == right:
            count3 += 1
print("锻炼预测准确度为：%f" %(count3/len(y_test3)))

y_train4 = data.iloc[0:9999,36].as_matrix() #训练样本标签列
x_train4 = data.iloc[0:9999,12:20].as_matrix() #训练样本特征
y_test4 = data.iloc[:,36].as_matrix() #测试样本标签列
x_test4 = data.iloc[:,12:20].as_matrix() #测试样本特征
clf = MultinomialNB().fit(x_train4,y_train4)
clf.predict(x_test4)
end4 = time.time()
print ("modeltime: %f s" %(end4-end3))
count4 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x_test4), y_test4):
      if left == right:
            count4 += 1
print("休闲预测准确度为：%f" %(count4/len(y_test4)))

y_train5 = data.iloc[0:9999,36].as_matrix() #训练样本标签列
x_train5 = data.iloc[0:9999,22:25].as_matrix() #训练样本特征
y_test5 = data.iloc[:,36].as_matrix() #测试样本标签列
x_test5 = data.iloc[:,22:25].as_matrix() #测试样本特征
clf = MultinomialNB().fit(x_train5,y_train5)
clf.predict(x_test5)
end5 = time.time()
print ("modeltime: %f s" %(end5-end4))
count5 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x_test5), y_test5):
      if left == right:
            count5 += 1
print("家庭情况预测准确度为：%f" %(count5/len(y_test5)))


end = time.time()
print ("endtime: %f s" %(end-start))
print ("Script ends...")