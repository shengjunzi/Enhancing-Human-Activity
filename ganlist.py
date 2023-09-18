import os
import random
def ganlist(gaf_class):
    train_ratio=0.7
    vali_ratio=0.5
    
    train=open('data/data141031/CSI-GAF'+str(gaf_class)+'/traindata.txt','w+')
    val=open('data/data141031/CSI-GAF'+str(gaf_class)+'/validata.txt','w+')
    test=open('data/data141031/CSI-GAF'+str(gaf_class)+'/testdata.txt','w+')
    root_dir='data/data141031/CSI-GAF'+str(gaf_class)
    train_dir='data'
    path=os.path.join(root_dir,train_dir)
    img_path=os.listdir(path)
    for line in img_path:
        if random.uniform(0, 1) < train_ratio: 
            train.writelines(line)
            train.write('\r\n')   
        else:
            if random.uniform(0, 1)< vali_ratio:
               val.writelines(line)
               val.write('\r\n')
            else:
                test.writelines(line)
                test.write('\r\n')
            
    train.close()
    val.close()
    test.close()