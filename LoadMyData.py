
import os
import cv2
import numpy as np
import paddle
import os
import cv2
import numpy as np
# import paddle.vision.transforms as T
import paddle.vision.transforms as T
class MyData(paddle.io.Dataset):
    def __init__(self,root_dir,class_dir,gaf_class):
        self.class_dir=class_dir
        self.root_dir=root_dir
        self.data=[]
        with open('data/data141031/CSI-GAF'+str(gaf_class)+'/{}.txt'.format(class_dir)) as f:
            for line in f.readlines():
                line=line.strip()
                label=int(line.split("-")[0])
                label=label-1
                if len(line) > 0:
                    self.data.append([line, label])#
        self.transforms = T.Compose([
               T.Resize(224)
            ])

    def __getitem__(self, item):
        img_name,label=self.data[item]
        img_item_path=os.path.join(self.root_dir,img_name)
        img = cv2.imread(img_item_path)
        img=self.transforms(img)
        # img = self.transforms(img)
        img=img.transpose(2,0,1)
        img = img.astype('float32')
        label=paddle.to_tensor(label)
        return img, label 

    def __len__(self):
        return len(self.data)