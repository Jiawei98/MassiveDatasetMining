# read in image information
image_info<-na.omit(read.csv("../dataset/exp_styles.csv"))
image_info<-image_info[,c("id","articleType")]
image_info$articleType<-str_remove_all(image_info$articleType, " ")
image_category_count<-table(image_info$articleType)
image_category_names<-names(image_category_count[image_category_count>10])
image_info<-image_info[image_info$articleType %in% image_category_names, ]
image_info<-image_info[order(image_info$articleType),]
rownames(image_info)<-NULL

## 60% train & 40% test
smp_size <- floor(0.6 * nrow(image_info))
set.seed(123)
train_ind <- sample(seq_len(nrow(image_info)), size = smp_size)
image_info$isTrain<-F
image_info$isTrain[train_ind]<-T

# create train/test folders
for (cur_category in image_category_names){
  dir.create(paste0("../dataset/images_train/", cur_category), recursive=T)
  dir.create(paste0("../dataset/images_test/", cur_category), recursive=T)
}

for (cur_num in 1:nrow(image_info)){
  filep<-paste0("../dataset/images_gray/", image_info$id[cur_num], ".jpg")
  category_name<-image_info$articleType[cur_num]
  folder_name<-ifelse(image_info$isTrain[cur_num], "images_train/", "images_test/")
  folderp<-paste0("../dataset/", folder_name, category_name)
  system2("cp", c(filep, folderp))
}

