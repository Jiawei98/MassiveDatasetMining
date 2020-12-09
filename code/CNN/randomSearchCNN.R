filter_num1=c(4,8,12)
kernel_size1=c(3,5)
filter_num2=c(24,32,48)
kernel_size2=c(3,5)
hidden_layer1=c(120,100,80)
hidden_layer2=c(60,50,40)
learning_rate=c(0.001,0.003,0.01,0.03)
batch_size=c(32,64,128,256)

set.seed(123)
n_rep<-1000

list_hp<-list(filter_num1, kernel_size1, filter_num2, kernel_size2, hidden_layer1, hidden_layer2, learning_rate, batch_size)
comb_hp<-data.frame(matrix(NA, n_rep, length(list_hp)))
colnames(comb_hp)<-c("filter_num1", "kernel_size1", "filter_num2", "kernel_size2", "hidden_layer1", "hidden_layer2", "learning_rate", "batch_size")

for (cur_row in 1:n_rep){
  for (cur_col in 1:length(list_hp)){
    cur_hp<-list_hp[[cur_col]]
    comb_hp[cur_row, cur_col]<-sample(cur_hp, 1)
  }
}

write.csv(comb_hp, "randomSearchCNN.csv", row.names=F)
