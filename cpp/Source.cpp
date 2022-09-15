#include <iostream>
#include <string>
#include <sstream>
#include <numeric>
#include <format>
#include <math.h>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\core\types.hpp>

#include <opencv2\imgproc.hpp>

#include "opencv2/imgproc/imgproc.hpp"

#include <fdeep/fdeep.hpp>
using namespace cv;

Mat src_gray;
Mat drawing;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void*);
double get_corner(Mat resized, int stt_chi_tiet, Point hesothamchieu, double hesogiam, Mat& arrImg);
Point midpoint(Point p1, Point p2);
double linalg_norm(Point p1, Point p2);

int main(int argc, const char** argv)
{
    std::stringstream stream;
    /*Mat image = imread("D:/New folder (5)/20212/do-an-20212/New folder (6)/runModelCpp/im1.png", IMREAD_GRAYSCALE);
    Mat resizedImg;
    resize(image, resizedImg, Size(28, 28));
    const auto mymodel = fdeep::load_model("D:/New folder (5)/20212/do-an-20212/New folder (6)/runModelCpp/my_model.json"); // load the converted model

    const auto input = fdeep::tensor_from_bytes(resizedImg.ptr(),
        static_cast<std::size_t>(resizedImg.rows),
        static_cast<std::size_t>(resizedImg.cols),
        static_cast<std::size_t>(resizedImg.channels()),
        0.0f, 1.0f);
    auto result = mymodel.predict({ input }); // predict the image's label and ouput a 1x2 tensor containing each class probability
    std::cout<<"TUng " << fdeep::show_tensors(result) << std::endl; // print the tensor
    cv::waitKey();*/

    const auto mymodel = fdeep::load_model("D:/New folder (5)/20212/do-an-20212/New folder (6)/runModelCpp/my_model.json"); // load the converted model
    Mat image = imread("D:/New folder (5)/20212/do-an-20212/New folder (6)/runModelCpp/im1.png", IMREAD_GRAYSCALE);
    auto input = fdeep::tensor_from_bytes(image.ptr(),
        static_cast<std::size_t>(28),
        static_cast<std::size_t>(28),
        static_cast<std::size_t>(1),
        0.0f, 1.0f);

    Mat dst1, dst2, dst3, dst4, dst5, dst6, dst7;
    Mat src = imread("D:/New folder (5)/20212/do-an-20212/New folder (6)/runModelCpp/image_27.png");//, IMREAD_GRAYSCALE);
    Mat cropped_image = src;// (Range(0, 30), Range(450, 640));

    cvtColor(src, cropped_image, COLOR_BGR2GRAY);
    cv::GaussianBlur(cropped_image, dst1, cv::Size(3, 3), 1);
    cv::threshold(dst1, dst2, 0, 127, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);

    cv::Canny(dst2, dst3, 30, 100);
    Mat canny_output;
    cv::threshold(dst3, canny_output, 127, 255, 0);

    const char* source_window = "Source";
    namedWindow(source_window);
    imshow(source_window, src);
    const int max_thresh = 255;

    drawing = src;

    vector<vector<Point> > contours;
    findContours(canny_output, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


    double area;
    double x, y, h, w;
    int C_x, C_y;
    Moments M;
    Mat small_img_2;
    Point hesothamchieu;
    Point C;
    cv::Rect rect;
    for (size_t i = 0; i < contours.size(); i++)
    {
        area = contourArea(contours[i]);
        rect = cv::boundingRect(contours[i]);
        if (area > 3000) {
            //drawContours(drawing, contours_poly, (int)i, color);
            rectangle(drawing, rect.tl(), rect.br(), cv::Scalar(255, 255, 255), 1);
            x = rect.tl().x;
            y = rect.tl().y;
            h = rect.height;
            w = rect.width;

            x = max(x, y);
            h = max(h, w);

            M = cv::moments(contours[i]);
            if (M.m00 != 0) {
                C_x = int(M.m10 / M.m00);
                C_y = int(M.m01 / M.m00);
            }
            else {
                cout << "co gi do sai sai" << endl;
            }
            hesothamchieu = Point(C_x - h / 2, C_y - h / 2);

            small_img_2 = cropped_image(Range(int(C_y - h / 2), int(C_y + h / 2)), Range(int(C_x - h / 2), int(C_x + h / 2)));
            imshow("Contours " + std::to_string(i), small_img_2);

            Mat resizedImg;
            resize(small_img_2, resizedImg, Size(28, 28));

            input = fdeep::tensor_from_bytes(resizedImg.ptr(),
                static_cast<std::size_t>(28),
                static_cast<std::size_t>(28),
                static_cast<std::size_t>(1),
                0.0f, 1.0f);
            auto result = mymodel.predict_class({ input });
            //predict the image's label and ouput a 1x2 tensor containing each class probability
            //std::cout << "TUng " + std::to_string(i) << fdeep::show_tensors(result) << std::endl; // print the tensor

            //double r5 = result.pop_back();
            std::cout << "ket qua: " << int(result) + 1 << std::endl;

            double hesogiam = 28 / h;
            double corner = get_corner(small_img_2, int(result) + 1, hesothamchieu, 1, drawing);

            Point org = Point(int(C_x - h / 2), int(C_y + h / 2) + 20);
            cv::putText(drawing, "ct " + std::to_string(int(result) + 1) + " Goc: " + std::to_string(corner), org, cv::FONT_HERSHEY_SIMPLEX,
                0.6, Scalar(255, 255, 0), 2, cv::LINE_AA);


        }
    }
    imshow("Contours", drawing);

    waitKey();

    return 0;
}

double get_corner(Mat resized, int stt_chi_tiet, Point hesothamchieu, double hesogiam, Mat& arrImg) {
    Mat blurred, img_threshold;
    cv::GaussianBlur(resized, blurred, cv::Size(5, 5), 3);
    cv::threshold(blurred, img_threshold, 100, 150, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
    vector<vector<Point> > contours;
    findContours(img_threshold, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    double area;
    Rect rect;
    Moments M;
    int C_x, C_y;
    vector<Point> k1;

    for (size_t i = 0; i < contours.size(); i++) {
        area = contourArea(contours[i]);
        if (area > 350) continue;
        if (area < 100) continue;
        rect = cv::boundingRect(contours[i]);
        if (rect.height < 40 && rect.width < 40) {
            rectangle(arrImg, rect.tl() + hesothamchieu, rect.br() + hesothamchieu, cv::Scalar(255, 0, 0), 1);
            M = cv::moments(contours[i]);
            if (M.m00 != 0) {
                C_x = int(M.m10 / M.m00);
                C_y = int(M.m01 / M.m00);
            }
            else {
                cout << "co gi do sai sai" << endl;
            }
            k1.push_back(Point(int(C_x / hesogiam), int(C_y / hesogiam)));
        }
    }

    for (Point x : k1) {
        cout << "fafa" << x << endl;
    }

    double corner = 0;
    double detal_x, detal_y;
    Point center;

    if (k1.size() == 2 & stt_chi_tiet == 2) {
        cout << "hinh tron nho: " << k1.size() << ". stt chi tiet: " << stt_chi_tiet << endl;
        center = midpoint(k1[0], k1[1]);


        for (int i = 0; i < k1.size(); i++) {
            if (k1[i].x >= center.x & k1[i].y >= center.y) {
                detal_x = k1[i].x - center.x;
                detal_y = k1[i].y - center.y;
                corner = atan(detal_y / detal_x) * 180 / 3.14;
                cout << "21" << endl;
                cout << "goc: " << corner << endl;
                break;
            }
        }
        if (corner == 0) {
            for (int i = 0; i < k1.size(); i++) {
                if (k1[i].x <= center.x & k1[i].y >= center.y) {
                    detal_x = center.x - k1[i].x;
                    detal_y = k1[i].y - center.y;
                    corner = atan(detal_x / detal_y) * 180 / 3.14 + 90;
                    cout << "22" << endl;
                    cout << "goc: " << corner << endl;
                    break;
                }
            }
        }
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[0] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[1] + hesothamchieu), Scalar(0, 255, 0), 2);

        cv::line(arrImg, Point(center + hesothamchieu), Point(center.x + 70, center.y) + hesothamchieu, Scalar(0, 0, 255), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(center.x, center.y - 70) + hesothamchieu, Scalar(0, 0, 255), 2);
    }
    else if (k1.size() == 3 & stt_chi_tiet == 1) {
        Moments M1 = cv::moments(Mat(k1));
        int C_x_1, C_y_1;
        if (M1.m00 != 0) {
            C_x_1 = int(M1.m10 / M1.m00);
            C_y_1 = int(M1.m01 / M1.m00);
            center = Point(C_x_1, C_y_1);
        }
        else {
            cout << "co gi do sai sai" << endl;
        }
        corner = 0;
        for (int i = 0; i < k1.size(); i++) {
            if (k1[i].x >= center.x & k1[i].y >= center.y) {
                detal_x = k1[i].x - center.x;
                detal_y = k1[i].y - center.y;
                corner = atan(detal_y / detal_x) * 180 / 3.14;
                cout << "11" << endl;
                cout << "goc: " << corner << endl;
                break;
            }
        }
        if (corner == 0) {
            for (int i = 0; i < k1.size(); i++) {
                if (k1[i].x <= center.x & k1[i].y >= center.y) {
                    detal_x = center.x - k1[i].x;
                    detal_y = k1[i].y - center.y;
                    corner = atan(detal_x / detal_y) * 180 / 3.14 + 90;
                    cout << "12" << endl;
                    cout << "goc: " << corner << endl;
                    break;
                }
            }
        }
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[0] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[1] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[2] + hesothamchieu), Scalar(0, 255, 0), 2);

        cv::line(arrImg, Point(center + hesothamchieu), Point(center.x + 70, center.y) + hesothamchieu, Scalar(0, 0, 255), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(center.x, center.y - 70) + hesothamchieu, Scalar(0, 0, 255), 2);
    }
    else if (k1.size() == 4 & stt_chi_tiet == 3) {
        vector<double> khoangCach = { linalg_norm(k1[0], k1[3]),
            linalg_norm(k1[1], k1[3]),
            linalg_norm(k1[2], k1[3]) };
        int diem_tuong_ung_3 = std::max_element(khoangCach.begin(), khoangCach.end()) - khoangCach.begin();
        center = midpoint(k1[3], k1[diem_tuong_ung_3]);
        for (int i = 0; i < k1.size(); i++) {
            if (k1[i].x >= center.x & k1[i].y >= center.y) {
                detal_x = k1[i].x - center.x;
                detal_y = k1[i].y - center.y;
                corner = atan(detal_y / detal_x) * 180 / 3.14;
                cout << "31" << endl;
                cout << "goc: " << corner << endl;
                break;
            }
        }
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[0] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[1] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[2] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[3] + hesothamchieu), Scalar(0, 255, 0), 2);

        cv::line(arrImg, Point(center + hesothamchieu), Point(center.x + 70, center.y) + hesothamchieu, Scalar(0, 0, 255), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(center.x, center.y - 70) + hesothamchieu, Scalar(0, 0, 255), 2);
    }
    else if (k1.size() == 4 && (stt_chi_tiet == 4 || stt_chi_tiet == 5)) {
        vector<double> khoangCach = { linalg_norm(k1[0], k1[3]),
            linalg_norm(k1[1], k1[3]),
            linalg_norm(k1[2], k1[3]) };
        int diem_tuong_ung_3 = std::max_element(khoangCach.begin(), khoangCach.end()) - khoangCach.begin();
        center = midpoint(k1[3], k1[diem_tuong_ung_3]);
        Point a11 = k1[diem_tuong_ung_3];
        Point a12 = k1[3];
        Point a21, a22;
        if (diem_tuong_ung_3 == 0) {
            a21 = k1[1];
            a22 = k1[2];
        }
        else if (diem_tuong_ung_3 == 1) {
            a21 = k1[0];
            a22 = k1[2];
        }
        else if (diem_tuong_ung_3 == 2) {
            a21 = k1[0];
            a22 = k1[1];
        }
        Point t11 = midpoint(a11, a21);
        Point t12 = midpoint(a11, a22);
        Point t21 = midpoint(a12, a21);
        Point t22 = midpoint(a12, a22);
        vector<Point> k2;

        if (linalg_norm(a11, a22) < linalg_norm(a11, a21)) {
            k2.push_back(t12);
            k2.push_back(t21);
            k2.push_back(t22);
            k2.push_back(t11);
        }
        else {
            k2.push_back(t11);
            k2.push_back(t22);
            k2.push_back(t21);
            k2.push_back(t12);
        }
        for (int i = 0; i < k2.size(); i++) {
            if (k2[i].x >= center.x && k2[i].y >= center.y) {
                detal_x = k2[i].x - center.x;
                detal_y = k2[i].y - center.y;
                corner = atan(detal_y / detal_x) * 180 / 3.14;
                cout << "4~5" << endl;
                if (i > 2) {
                    corner = corner + 90;
                }
                cout << "goc: " << corner << endl;
                break;
            }
        }
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[0] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[1] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[2] + hesothamchieu), Scalar(0, 255, 0), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(k1[3] + hesothamchieu), Scalar(0, 255, 0), 2);

        cv::line(arrImg, Point(center + hesothamchieu), Point(center.x + 70, center.y) + hesothamchieu, Scalar(0, 0, 255), 2);
        cv::line(arrImg, Point(center + hesothamchieu), Point(center.x, center.y - 70) + hesothamchieu, Scalar(0, 0, 255), 2);
    }


    //imshow("all", arrImg);
    return corner;

}

Point midpoint(Point p1, Point p2) {
    return Point(int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2));
}

double linalg_norm(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void thresh_callback(int, void*)
{
    Mat canny_output;
    Canny(src_gray, canny_output, thresh, thresh * 2);
    vector<vector<Point> > contours;
    findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Point2f>centers(contours.size());
    vector<float>radius(contours.size());

    /*for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = boundingRect(contours_poly[i]);
        minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
    }*/
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        //drawContours(drawing, contours_poly, (int)i, color);
        rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(), 2);
        //circle(drawing, centers[i], (int)radius[i], color, 2);
    }
    imshow("Contours", drawing);
}
