options(scipen = 200)

# hyperparameters：
filter_num1=c(4,8,12,16)
kernel_size1=c(3,5)
filter_num2=c(32,48,64,80)
kernel_size2=c(3,5)
hidden_layer1=c(120,160,200)
hidden_layer2=c(20,40,60)
learning_rate=c(0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01)
batch_size=c(16,32,64,128,256,512)

# get 3000 different combinations
set.seed(123)
n_rep<-3200  # make sure to get more than 3000 unique combinations

list_hp<-list(filter_num1, kernel_size1, filter_num2, kernel_size2, hidden_layer1, hidden_layer2, learning_rate, batch_size)
comb_hp<-data.frame(matrix(NA, n_rep, length(list_hp)))
colnames(comb_hp)<-c("filter_num1", "kernel_size1", "filter_num2", "kernel_size2", "hidden_layer1", "hidden_layer2", "learning_rate", "batch_size")

for (cur_row in 1:n_rep){
  for (cur_col in 1:length(list_hp)){
    cur_hp<-list_hp[[cur_col]]
    comb_hp[cur_row, cur_col]<-sample(cur_hp, 1)
  }
}

comb_hp<-unique(comb_hp)[1:3000,]

write.csv(comb_hp, "randomSearchCNN.csv", row.names=F)
