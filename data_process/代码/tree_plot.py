#-*- coding: utf-8 -*-
#生成决策树图
from sklearn import tree
import pandas as pd
from sklearn.tree import export_graphviz
import sys
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

data_file = 'data.xls'
data = pd.read_excel(data_file)
    
y = data.iloc[:500,62]
x = data.iloc[:500,0:46]
model = tree.DecisionTreeClassifier()
clf = model.fit(x,y)

with open("tree.dot",'w') as f:
    f = tree.export_graphviz(clf,out_file=f)
#dot -Tpdf tree.dot -o tree.pdf
#在命令行执行本行代码