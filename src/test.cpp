#include "../include/line2Dup.h"
#include "../include/utils_.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
// #include <opencv2/dnn.hpp>
using namespace std;
using namespace cv;

static std::string prefix = "/home/seanyan/projects/shape_based_matching/test/";

namespace  cv_dnn {
    namespace
    {

        template <typename T>
        static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                const std::pair<float, T>& pair2)
        {
            return pair1.first > pair2.first;
        }

    } // namespace

    inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                        std::vector<std::pair<float, int> >& score_index_vec)
    {
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] > threshold)
            {
                score_index_vec.push_back(std::make_pair(scores[i], i));
            }
        }
        std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                        SortScorePairDescend<int>);
        if (top_k > 0 && top_k < (int)score_index_vec.size())
        {
            score_index_vec.resize(top_k);
        }
    }

    template <typename BoxType>
    inline void NMSFast_(const std::vector<BoxType>& bboxes,
        const std::vector<float>& scores, const float score_threshold,
        const float nms_threshold, const float eta, const int top_k,
        std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
    {
        CV_Assert(bboxes.size() == scores.size());
        std::vector<std::pair<float, int> > score_index_vec;
        GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

        // Do nms.
        float adaptive_threshold = nms_threshold;
        indices.clear();
        for (size_t i = 0; i < score_index_vec.size(); ++i) {
            const int idx = score_index_vec[i].second;
            bool keep = true;
            for (int k = 0; k < (int)indices.size() && keep; ++k) {
                const int kept_idx = indices[k];
                float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            }
            if (keep)
                indices.push_back(idx);
            if (keep && eta < 1 && adaptive_threshold > 0.5) {
                adaptive_threshold *= eta;
            }
        }
    }


    // copied from opencv 3.4, not exist in 3.0
    template<typename _Tp> static inline
    double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
        _Tp Aa = a.area();
        _Tp Ab = b.area();

        if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
            // jaccard_index = 1 -> distance = 0
            return 0.0;
        }

        double Aab = (a & b).area();
        // distance = 1 - jaccard_index
        return 1.0 - Aab / (Aa + Ab - Aab);
    }

    template <typename T>
    static inline float rectOverlap(const T& a, const T& b)
    {
        return 1.f - static_cast<float>(jaccardDistance__(a, b));
    }

    void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                            const float score_threshold, const float nms_threshold,
                            std::vector<int>& indices, const float eta=1, const int top_k=0)
    {
        NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
    }
}    


void circle_gen(){
    Mat bg = Mat(800, 800, CV_8UC3, {0, 0, 0});
    cv::circle(bg, {400, 400}, 200, {255,255,255}, -1);
    cv::imshow("test", bg);
    waitKey(0);
}

int Mouse::mouse_event;
int Mouse::mouse_x;
int Mouse::mouse_y;

void detection(string detect_mode, string mode = "test"){
    
//    mode = "none";
    if(detect_mode == "Line2D")
    {
        linemod::Detector detector(80, {5, 8}, detect_mode);
        if(mode == "train"){
            Mat src = imread(prefix+"case6/rgb.png");
            // Mat mask = Mat(img.size(), CV_8UC1, {255});
            Mat mask = imread(prefix+"case6/mask.png");

            Mat src_rgb, src_mask;
            Size roi_size(60, 60);
            // int Mouse::m_event;
            // int Mouse::m_x;
            // int Mouse::m_y;
            namedWindow("rgb");
            Mouse::start("rgb");

            while(1){
                Point mouse(Mouse::x(), Mouse::y());
                int event = Mouse::event();
                Point roi_offset(roi_size.width / 2, roi_size.height / 2);
                Point pt1 = mouse - roi_offset; // top left
                Point pt2 = mouse + roi_offset; // bottom right
                Rect roi = Rect(pt1.x, pt1.y, roi_size.width, roi_size.height);

                if(event == cv::EVENT_RBUTTONDOWN)
                {
                    // Compute object mask by subtracting the plane within the ROI
                    Mat Roi = src(roi);
                    Mat Roi_mask = mask(roi);
                    Roi.copyTo(src_rgb);
                    src_mask = Mat(src_rgb.size(), CV_8UC1, {255});
                    destroyAllWindows();
                    break;
                }

                Mat display = src.clone();
                cv::rectangle(display, pt1, pt2, CV_RGB(0,0,0), 3);
                cv::rectangle(display, pt1, pt2, CV_RGB(255,255,0), 1);
                imshow("rgb", display);
                char key = waitKey(10);
                if( key == 'q' )
                    break;
            }
            shape_based_matching::shapeInfo shapes(src_rgb, src_mask);
            shapes.angle_range = {0, 360};
            shapes.angle_step = 1;
            shapes.scale_range = {0.8, 1.2};
            shapes.scale_step = 0.1;
            shapes.produce_infos();     // 生成形状信息
            std::vector<shape_based_matching::shapeInfo::shape_and_info> infos_have_templ;
            string class_id = "Ape";
            for(auto& info: shapes.infos){
                imshow("train", info.src);
                waitKey(1);

                std::cout << "\ninfo.angle: " << info.angle << std::endl;
                int templ_id = detector.addTemplate(info.src, class_id, info.mask, info.angle, info.scale);
                std::cout << "templ_id: " << templ_id << std::endl;
                if(templ_id != -1){
                    infos_have_templ.push_back(info);
                }
            }
            detector.writeClasses(prefix+"case6/%s_templ.yaml");
            shapes.save_infos(infos_have_templ, shapes.src, shapes.mask, prefix + "case6/test_info.yaml");
            std::cout << "train end" << std::endl;
        }else if(mode=="test"){
            std::vector<std::string> ids;
            ids.push_back("Ape");
            detector.readClasses(ids, prefix+"case6/%s_templ.yaml");
            Mat test_img = imread(prefix+"case6/1_rgb.png");

            int stride = 16;
            int n = test_img.rows/stride;
            int m = test_img.cols/stride;
            Rect roi(0, 0, stride*m , stride*n);

            test_img = test_img(roi).clone();
            std::vector< Mat > detect;
            detect.push_back(test_img);

            Timer match_timer;
            match_timer.start();
            auto matches = detector.match(detect, 90, ids);
            match_timer.stop();
            // one output match:
            // x: top left x
            // y: top left y
            // template_id: used to find templates
            // similarity: scores, 100 is best


            std::cout << "matches.size(): " << matches.size() << std::endl;
            size_t top5 = 500;
            if(top5>matches.size()) top5=matches.size();

            vector<Rect> boxes;
            vector<float> scores;
            vector<int> idxs;
            for(auto match: matches){
                Rect box;
                box.x = match.x;
                box.y = match.y;

                auto templ = detector.getTemplates("Ape",
                                                match.template_id);

                box.width = templ[0].width;
                box.height = templ[0].height;
                boxes.push_back(box);
                scores.push_back(match.similarity);
            }
            cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

            for(auto idx: idxs){
                auto match = matches[idx];
                auto templ = detector.getTemplates("Ape",
                                                match.template_id);

                int x = templ[0].width + match.x;
                int y = templ[0].height + match.y;
                int r = templ[0].width/2;
                cv::Vec3b randColor;
                randColor[0] = rand()%155 + 100;
                randColor[1] = rand()%155 + 100;
                randColor[2] = rand()%155 + 100;

                std::ostringstream angle;
                std::ostringstream scale;
                angle << match.angle;
                scale << match.scale;

                for(int i=0; i<templ[0].features.size(); i++){
                    auto feat = templ[0].features[i];
                    cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
                }

                cv::putText(test_img, to_string(int(round(match.similarity)))+" "+ angle.str()+" "+ scale.str(),
                            Point(match.x+r, match.y-3), FONT_HERSHEY_PLAIN, 1, randColor);

                cv::rectangle(test_img, {match.x, match.y}, {x, y}, randColor, 2);

                std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
                std::cout << "match.similarity: " << match.similarity << std::endl;
            }

            imshow("img", test_img);
            waitKey(0);

            std::cout << "Line2D test end" << std::endl;
        }
    }
    else if(detect_mode == "LineMod"){
        cv::Ptr<linemod::Detector> detector;
        string filename = prefix + "case6/linemod_Ape_templates.yml";

        if(mode == "train")
        {
            Mat src = imread(prefix+"case6/rgb.png");
            Mat mask = imread(prefix+"case6/mask.png");
            Mat depth = imread(prefix+"case6/depth.png");
            detector = linemod::getDefaultLINEMOD(detect_mode);

            Mat src_rgb, src_depth, src_mask;
            Size roi_size(60, 60);
            // int Mouse::m_event;
            // int Mouse::m_x;
            // int Mouse::m_y;
            namedWindow("rgb");
            Mouse::start("rgb");

            while(1){
                Point mouse(Mouse::x(), Mouse::y());
                int event = Mouse::event();
                Point roi_offset(roi_size.width / 2, roi_size.height / 2);
                Point pt1 = mouse - roi_offset; // top left
                Point pt2 = mouse + roi_offset; // bottom right
                Rect roi = Rect(pt1.x, pt1.y, roi_size.width, roi_size.height);

                if(event == cv::EVENT_RBUTTONDOWN)
                {
                    // Compute object mask by subtracting the plane within the ROI
                    Mat Roi = src(roi);
                    Mat Roi_mask = mask(roi);
                    Mat Roi_depth = depth(roi);
                    Roi.copyTo(src_rgb);
                    Roi_depth.copyTo(src_depth);
                    src_mask = Mat(src_rgb.size(), CV_8UC1, {255});
                    destroyAllWindows();
                    break;
                }

                Mat display = src.clone();
                cv::rectangle(display, pt1, pt2, CV_RGB(0,0,0), 3);
                cv::rectangle(display, pt1, pt2, CV_RGB(255,255,0), 1);
                imshow("rgb", display);
                char key = waitKey(10);
                if( key == 'q' )
                    break;
            }

            shape_based_matching::shapeInfo shapes(src_rgb, src_mask);
            shapes.angle_range = {0, 360};
            shapes.angle_step = 1;
            shapes.scale_range = {0.8, 1.2};
            shapes.scale_step = 0.1;
            shapes.produce_infos();     // 生成形状信息
            shape_based_matching::shapeInfo depths(src_depth, src_mask);
            depths.angle_range = {0, 360};
            depths.angle_step = 1;
            depths.scale_range = {0.8, 1.2};
            depths.scale_step = 0.1;
            depths.produce_infos();     // 生成深度信息

            std::vector<shape_based_matching::shapeInfo::shape_and_info> infos_have_templ;
            string class_id = "Ape";
            std::vector< Mat > temp;
            for(int i = 0; i < shapes.infos.size(); i++){
                // imshow("train", shapes.infos[i].src);
                // waitKey(1);
                temp.push_back(shapes.infos[i].src);
                temp.push_back(depths.infos[i].src);
                std::cout << "\ninfo.angle: " << shapes.infos[i].angle << std::endl;
                std::cout << "\ninfo.scale: " << shapes.infos[i].scale << std::endl;
                int templ_id = detector->addTemplate(temp, class_id, shapes.infos[i].mask, shapes.infos[i].angle, shapes.infos[i].scale);
                temp.pop_back();
                temp.pop_back();
                std::cout << "templ_id: " << templ_id << std::endl;
                if(templ_id != -1){
                    infos_have_templ.push_back(shapes.infos[i]);
                }
            }

            writeLinemod(detector, filename);
            printf("Wrote detector and templates to %s\n", filename.c_str());
            shapes.save_infos(infos_have_templ, shapes.src, shapes.mask, prefix + "case6/linemod_info.yaml");
            std::cout << "train end" << std::endl;
        }else if(mode=="test"){

            std::vector<std::string> ids;
            ids.push_back("Ape");
            detector = readLinemod(filename, detect_mode);
            cout << detector->detect_mode << endl;
            Mat test_rgb = imread(prefix+"case6/1_rgb.png");
            Mat test_depth = imread(prefix+"case6/1_depth.png");

            int stride = 16;
            int n = test_rgb.rows/stride;
            int m = test_rgb.cols/stride;
            Rect roi(0, 0, stride*m , stride*n);

            test_rgb = test_rgb(roi).clone();
            test_depth = test_depth(roi).clone();

            std::vector< Mat > detect;
            detect.push_back(test_rgb);
            detect.push_back(test_depth);
            Timer match_timer;
            match_timer.start();
            auto matches = detector->match(detect, (float)85, ids);
            match_timer.stop();
            // one output match:
            // x: top left x
            // y: top left y
            // template_id: used to find templates
            // similarity: scores, 100 is best


            std::cout << "matches.size(): " << matches.size() << std::endl;
            size_t top5 = 500;
            if(top5>matches.size()) top5=matches.size();

            vector<Rect> boxes;
            vector<float> scores;
            vector<int> idxs;
            for(auto match: matches){
                Rect box;
                box.x = match.x;
                box.y = match.y;

                auto templ = detector->getTemplates("Ape",
                                                match.template_id);

                box.width = templ[0].width;
                box.height = templ[0].height;
                boxes.push_back(box);
                scores.push_back(match.similarity);
            }
            cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

            for(auto idx: idxs){
                auto match = matches[idx];
                auto templ = detector->getTemplates("Ape",
                                                match.template_id);

                int x = templ[0].width + match.x;
                int y = templ[0].height + match.y;
                int r = templ[0].width/2;
                cv::Vec3b randColor;
                randColor[0] = rand()%155 + 100;
                randColor[1] = rand()%155 + 100;
                randColor[2] = rand()%155 + 100;

                std::ostringstream angle;
                std::ostringstream scale;
                angle << match.angle;
                scale << match.scale;

                for(int i=0; i<templ[0].features.size(); i++){
                    auto feat = templ[0].features[i];
                    cv::circle(test_rgb, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
                }

                cv::putText(test_rgb, to_string(int(round(match.similarity)))+" "+ angle.str()+" "+ scale.str(),
                            Point(match.x+r, match.y-3), FONT_HERSHEY_PLAIN, 1, randColor);

                cv::rectangle(test_rgb, {match.x, match.y}, {x, y}, randColor, 2);

                std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
                std::cout << "match.similarity: " << match.similarity << std::endl;
            }

            imshow("img", test_rgb);
            waitKey(0);

            std::cout << "LineMod test end" << std::endl;
        }
    }
}
int main(){
    detection("LineMod");           ///  detect_mode include:  "Line2D", "LineMod"
    return 0;
}
