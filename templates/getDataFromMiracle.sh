miracle='miracle'
directory='/home/z1143051/cuda/MultiAgentSystem'
archive='data.tar.bz2'

mkdir -p data

scp $miracle:$directory/$archive ./data/
tar -jxvf ./data/$archive -C ./data
rm ./data/$archive

