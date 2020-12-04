# install python 3
tar -xzf python38.tar.gz &&
export PATH=$PWD/python/bin:$PATH && 
python3 --version &&

# install packages 
mkdir packages && 
python3 -m pip install --target=$PWD/packages numpy pandas torch torchvision Pillow &&

## pack files 
tar -czf packages.tar.gz packages/