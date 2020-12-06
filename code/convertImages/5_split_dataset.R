require("stringr")

set.seed(123)
# read in image information
image_filenames<-list.files("images_compressed_tgz/")
image_info<-na.omit(read.csv("styles.csv"))
image_info<-image_info[,c("id", "articleType")]
image_info$id<-str_pad(image_info$id, 5, pad = "0")
image_info<-image_info[paste0(image_info$id, ".jpg") %in% image_filenames, ]
image_info$articleType<-str_remove_all(image_info$articleType, " ")
image_category_count<-table(image_info$articleType)
image_category_names<-names(image_category_count[image_category_count>500])
image_info<-image_info[image_info$articleType %in% image_category_names, ]
image_info<-image_info[order(image_info$articleType), ]
rownames(image_info)<-NULL

## 60% train, 30% validation & 10% test
train_size <- floor(0.6 * nrow(image_info))
train_ind <- sample(seq_len(nrow(image_info)), size = train_size)
image_info$isTrain<-F
image_info$isTrain[train_ind]<-T

validation_size <- floor(0.3 * nrow(image_info))
validation_ind <- sample(which(!image_info$isTrain), size = validation_size)
image_info$isValidation<-F
image_info$isValidation[validation_ind]<-T

# create train/validation/test folders
for (cur_category in image_category_names){
  dir.create(paste0("dataset/images_train/", cur_category), recursive=T)
  dir.create(paste0("dataset/images_validation/", cur_category), recursive=T)
  dir.create(paste0("dataset/images_test/", cur_category), recursive=T)
}

for (cur_num in 1:nrow(image_info)){
  filep<-paste0("images_compressed_tgz/", image_info$id[cur_num], ".jpg")
  category_name<-image_info$articleType[cur_num]
  folder_name<-ifelse(image_info$isTrain[cur_num], "images_train/", ifelse(image_info$isValidation[cur_num], "images_validation/",  "images_test/"))
  folderp<-paste0("dataset/", folder_name, category_name)
  system2("cp", c(filep, folderp))
}

