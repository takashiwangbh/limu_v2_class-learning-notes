
#深度学习本地环境的安装

###使用conda/miniconda环境

### 
```vim
conda env remove d2l-zh
conda create -n -y d2l-zh python=3.8 pip
conda activate d2l-zh
```

###安装所需要的包
```vim
pip install -y jupyter d2l torch torchvision
```


