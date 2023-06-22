import os

class CustomDataset:
    #data load
    def __init__(self, path):
        # if mode =='train':     
        #     file_list = os.listdir(path)    #파일 경로 받기
        self.a = [1,2,3,4]
        pass

    #data preocessing
    def __getitem__(self, index):       #사용자가 직접 호출할 수 없다 -> 인자 변경 X
        self.a[index] +=2
        return self.a[index]

    #data return
    def __len__(self):
        return len(self.a)
    
    def __add__(self,a):
        return self.a[0]+1


#class 불러오기
dataset_inst = CustomDataset('/data/face1.jpg')
for i in dataset_inst:
    print(i)



print(len(dataset_inst))
#print(dataset_inst.__len__())

#변수 가져오기
print(dataset_inst.a)

#__add__ 매소드를 추가해서 가능해짐
print(dataset_inst+1)