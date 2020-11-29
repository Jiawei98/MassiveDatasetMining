# convert images

# doc of package https://rdrr.io/cran/OpenImageR/
if (!require("OpenImageR")) {
  install.packages("OpenImageR")
} 
img.list = list.files('../dataset/images')
head(img.list)



for (img.name in img.list[1:500]){
  print(img.name)
  img = readImage(paste('../dataset/images/', img.name, sep=''))
  img = rgb_2gray(img)
  img = resizeImage(img, 32, 32)
  writeImage(img, paste('../dataset/images_gray/', img.name, sep=''))
}


img.csv = data.frame(fileName=character(0), image=character(0))

for (img.name in img.list[1:500]){
  img = readImage(paste('../dataset/images_gray/', img.name, sep=''))
  img = paste(as.vector(img), collapse = ', ')
  img.csv = rbind(img.csv, data.frame(fileName=img.name, image = img) )
}

write.csv(img.csv, '../dataset/csv/gray_500.csv')


## to parse saved image in csv
a = img.csv[50,2] # read the value of the pixel array 
b = as.numeric(strsplit(a, ", ")[[1]]) # parse pixel array
c = matrix(b,  nrow = 32, ncol = 32) # convert array to matrix
imageShow(c)


