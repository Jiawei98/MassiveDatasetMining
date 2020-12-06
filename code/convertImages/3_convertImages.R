rm(list=ls())
arguments = (commandArgs(trailingOnly=TRUE))
fn = arguments[1]

# doc of package https://rdrr.io/cran/OpenImageR/
if (!require("OpenImageR")) {
  install.packages("OpenImageR")
} 
img.list = list.files("images")

cur_num=0
for (img.name in img.list){
  img = readImage(paste0("images", '/', img.name))
  img = rgb_2gray(img)
  img = resizeImage(img, 32, 32)
  writeImage(img, paste0(fn, '_gray/', img.name))
  cur_num=cur_num+1
  if (cur_num %% 100==0){print(cur_num)}
}
