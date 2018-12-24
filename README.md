# face_recognize
use mtcnn detect face and mobilefacenet calculate similarity

mkdir build && cd build  
cmake ..  
make -j  
./face_recognize  


有个问题是，当改变mtcnn的minsize时，同一张人脸相似度差异比较大，具体原因还在探索。
