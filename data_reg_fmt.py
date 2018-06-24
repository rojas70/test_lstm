datasets = joblib.load('datasets_reg.pkl')  
print('length of dataset:', len(datasets))
print('\n')

reg_fmt_datasets = []
for i, dataset in enumerate(datasets):
  print('length of dataset:', str(i), len(dataset))
  
  reg_fmt_dataset = []
  for j, sample in enumerate(dataset):
    left_hand = sample['left_hand']
    reg_fmt_dataset.append(left_hand)
      
    # print('length of sample:', str(j), len(sample))
    # print('length of left_hand:', str(j), left_hand.shape) 
  print('length of reg_fmt_dataset:', len(reg_fmt_dataset))  
  
  reg_fmt_datasets.append(reg_fmt_dataset)  

print('\n')  
print('length of reg_fmt_datasets:', len(reg_fmt_datasets))  


pickle.dump(reg_fmt_datasets, open("reg_fmt_datasets.pkl", "w"))