# sparse-representation
coding a query sample as a sparse linear combination of all training samples and then classifying it by evaluating which class leads to the minimal coding residual

clear all
clc
close all

database=[pwd '\ORL'];%使用的人脸库
train_samplesize=5;

address=[database '\s'];

rows=112;
cols=92;
ClassNum=40;
tol_num=10;
image_fmt='.bmp';
dim=[10 20 30 40 50 70 80 90];
recog_rate=zeros(1,length(dim));

%------------------------PCA降维
train=1:train_samplesize;
test=train_samplesize+1:tol_num;

train_num=length(train);
test_num=length(test);

[train_sample,train_label]=readsample(address,ClassNum,train,rows,cols,image_fmt);
[test_sample,test_label]=readsample(address,ClassNum,test,rows,cols,image_fmt);
m=1;
for pro_dim=dim
    [Pro_Matrix,Mean_Image]=my_pca(train_sample,pro_dim);
    train_pro=Pro_Matrix'*train_sample;
    test_pro=Pro_Matrix'*test_sample;
    
    %单位化
    train_norm=normc(train_pro);
    test_norm=normc(test_pro);
    
    recog_rate(m)=computaccuracy(train_norm,ClassNum,train_label,test_norm,test_label);
    m=m+1;
end
fprintf('每类训练样本个数为:%d\n',train_samplesize);
fprintf('维数及对应识别率为:\n');
disp([dim;recog_rate])

