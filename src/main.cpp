#include <opencv2/opencv.hpp>
#include "mobilefacenet.h"
#include "mtcnn.h"

using namespace std;
using namespace cv;

MtcnnDetector mtcnn("../models-mtcnn");
Recognize recognize("../models-mobilefacenet/");

Mat getwarpAffineImg(Mat &src, vector<Point2f> &landmarks)
{
    Point2f eyesCenter = Point2f((landmarks[0].x + landmarks[1].x) * 0.5f, (landmarks[0].y + landmarks[1].y) * 0.5f);

    double dy = (landmarks[1].y - landmarks[0].y);
    double dx = (landmarks[1].x - landmarks[0].x);
    double angle = atan2(dy, dx) * 180.0 / CV_PI;

    Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, 1.0);
    Mat rot;

    warpAffine(src, rot, rot_mat, src.size());
    return rot;
}

Mat mt(Mat frame)
{
    Mat face_area, affined_area;

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
    vector<FaceInfo> finalBbox = mtcnn.Detect(ncnn_img);

    const int num_box = finalBbox.size();
    vector<Rect> bbox;
    vector<Point2f> point;
    bbox.resize(num_box);
    for (int i = 0; i < num_box; i++) {
        bbox[i] = Rect(finalBbox[i].x[0], finalBbox[i].y[0], finalBbox[i].x[1] - finalBbox[i].x[0], finalBbox[i].y[1] - finalBbox[i].y[0]);
    }

    point.push_back(Point2f(finalBbox[0].landmark[0], finalBbox[0].landmark[1]));
    point.push_back(Point2f(finalBbox[0].landmark[2], finalBbox[0].landmark[3]));

    affined_area = getwarpAffineImg(frame, point);
    face_area = affined_area(bbox[0]).clone();

    return face_area;
}

vector<float> extract_feature(Mat img)
{
    vector<float> feature;
    recognize.start(img, feature);

    return feature;
}

int main()
{
    Mat img1 = imread("../gao1.jpg");
    Mat img2 = imread("../gao3.jpg");

    Mat face1 = mt(img1);
    Mat face2 = mt(img2);

    imshow("face1", face1);
    imshow("face2", face2);

    clock_t start, finish;
    start = clock();
    vector<float> feature1 = extract_feature(face1);
    finish = clock();
    cout << "mobilefacenet cost " << (float)(finish - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;
   
    start = clock();
    vector<float> feature2 = extract_feature(face2);
    finish = clock();
    cout << "mobilefacenet cost " << (float)(finish - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    cout << "Similar is " << calculSimilar(feature1, feature2) << endl;

    waitKey(0);

    return 0;
}
