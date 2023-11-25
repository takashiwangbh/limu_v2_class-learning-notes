# 数据操作    


N维数组是机器学习和神经网络的主要数据结构

#### 创建数组需要
* 定义数组形状
* 定义元素的数据类型
* 定义元素的值

#### 导入torch
虽然被称为pytorch，但是导入是导入torch；

```python
import torch
```

#### 张量表示数值形成的数组

```python
x = torch.arange(12)
x
```

#### shape:访问张量的形状和张量中的元素的总数

```python
x.shape
x.numel()
```

#### reshape：改变一个张量的形状而不改变数值和元素值

```python
X = x.reshape(3,4)
X
```

#### zeros：创造全0的张量

```python
torch.zeros((2,3,4))
```

#### ones: 创建全1的张量

```python
torch.ones((2,3,4))
```

#### 通过提供包含数值的列表，来为所需张量中每个元素赋予确定值
例如创建一个三维数组：

```python
torch.tensor([[[2,1,4,3],[1,2,3,4],[4,4,3,2]]])
```
#### 对张量做元素计算

```python
x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
x+y, x-y, x/y, x*y, x**y #**运算符是求幂运算
```

#### 可以把多个张量连结在一起

```python
x = torch.arange(12,dtype=torch.float32).reshape((3,4)) #生成一个0-11长度为12的张量，类型为float32，3行4列的二维张量
y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
torch.cat((x,y),dim=0),torch.cat((x,y),dim=1) #dim=0的时候按行拼接，dim=1的时候按列拼接
```

#### 通过逻辑运算符构建二元张量

```python
x = ([[1,1,1,1],[2,3,4,5]])
y = torch.arange(3).reshape((1,4))
x == y
```

#### 即使形状不同（维度得相同）仍然可以通过调用广播机制（broadcasting mechanism）来执行按元素操作

```python
a = torch.arange(3).reshape((3,1))
b = torch.arange(3).reshape((1,2))
a,b,a+b 
```

#### 可以用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素

```python
X[-1], X[1:3]
```

#### 通过索引来将元素写入矩阵

```python
X[1,2] = 9 #计算机是从0开始计数的，所以[1,2]是第二行第三列
X
```

#### 也可以对多个元素赋值相同的值

```python
X[0:2, : ] = 12 #将第一行和第三行，所有列的元素都赋值为12
X
```

#### 运行一些操作可能导致为新结果分配内存

```python
before = id(y)
y = y + x
id(y) == before
```
输出结果为false，因为之前的y的内存已经被重新分配给后面的y了。

#### 可以执行一些原地操作

```python
z = torch.zeros_like(y)#跟y的形状和类型一样，但是数据为0
z[ : ] = x + y #z的内存没有发生变化
```

#### 如果后续计算没有重复使用，也可以使用y[ : ] = x + y or x += y来减少操作的内存开销

```python
before = id(y)
y += x #or y[ : ] 
id(y) == before
```
输出为true

#### 转换为NumPy张量

```python
A = X.numpy()
B = torch.tensor(A)
type(A),type(B)
```
输出为（numpy.ndarray,torch.Tensor）

#### 将大小为1的张量转换为Python标量

```python
a = torch.tensor([3.5])
a,a.item(),float(a),int(a)
```
输出为（tensor([3.50000]）,3.5,3.5,3)

  
    

# 数据预处理
 
有一个原始数据，怎么读取进来使得通过机器学习的方法能够处理。

#### 创建一个人工数据集，并存储在csv文件

```python
#创建一个文件夹，文件名叫house_tiny.csv文件
#csv文件：每一行是一个数据，每一个域用逗号分开
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price/n') #列名
    f.write('NA,Pave,127500/n') #每行表示一个数据样本
    f.write('2,NA,10600/n')
    f.write('NA,NA,14000/n')
```

#### 从创建的csv文件中加载原始数据集

加载csv文件需要pandas库，如果没有需要：

```python
 !pip install pandas
```
当有了pandas库了以后，从csv文件中加载数据集：

```python
import pandas as pd
data = pd.read_csv(data_file)
print(data)
```
#### 插值：将数据改写或者插入

```python
inputs,outputs = data.iloc[ : ,0:2],data.iloc[ : , 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

#### 当input和output中所有的条目都是数值类型，可以转换为张量格式

```python
import torch
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
x,y 
```
  
    
            

# Q and A
 Q：数组跟不上
 A：学习Numpy
 Q：怎么快速区分维度
 A：
 ```python
 a.ndim #可以获取维度
```
