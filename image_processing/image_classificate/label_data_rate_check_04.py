import os
import matplotlib.pyplot as plt
PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'

# 각 라벨 별 카테고리 비율
class DataVisulizer:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.all_cnt = {}      #라벨별 몇개인지 보기위해
        self.train_cnt = {}
        self.val_cnt = {}
        self.test_cnt={}

    def load_data(self):
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'validation')
        test_dir = os.path.join(self.data_dir, 'test')


        # Train
        for label in os.listdir(train_dir):
            label_dir = os.path.join(train_dir, label)
            count = len(os.listdir(label_dir))
            self.train_cnt[label] = count
            self.all_cnt[label] = count
        
        # validation
        for label in os.listdir(val_dir):
            label_dir = os.path.join(val_dir,label)
            count = len(os.listdir(label_dir))
            if label in self.all_cnt:
                self.all_cnt[label] += count
                self.val_cnt[label] = count
            else:
                self.val_cnt[label] = count

        # Test
        for label in os.listdir(test_dir):
            label_dir = os.path.join(test_dir,label)
            count = len(os.listdir(label_dir))
            if label in self.all_cnt:
                self.all_cnt[label] += count
                self.test_cnt[label] = count
            else:
                self.test_cnt[label] = count

        #print(self.all_cnt)

    def visualize_data(self):
        labels = list(self.all_cnt.keys())
        counts = list(self.all_cnt.values())
        print(labels)
        # print(counts)

        plt.figure(figsize=(10,6))
        plt.bar(labels,counts)
        plt.title("Label data counts")
        plt.xlabel('Lables')
        plt.ylabel("Number of data")
        plt.xticks(rotation = 45, ha='right', fontsize=8)
        plt.show()


if __name__ == '__main__':
    
    test = DataVisulizer(f"{PATH}/food_dataset")
    test.load_data()
    test.visualize_data()