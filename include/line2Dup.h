#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <map>
#include <iostream>
// #include "utils_.h"

using namespace std;

// class Timer
// {
//     public:
//         Timer() : beg_(clock_::now()) {}
//         void reset() { beg_ = clock_::now(); }
//         double elapsed() const {
//             return std::chrono::duration_cast<second_>
//                 (clock_::now() - beg_).count(); }
//         void out(std::string message = ""){
//             double t = elapsed();
//             std::cout << message << "  elasped time:" << t << "s" << std::endl;
//             reset();
//         }
//     private:
//         typedef std::chrono::high_resolution_clock clock_;
//         typedef std::chrono::duration<double, std::ratio<1> > second_;
//         std::chrono::time_point<clock_> beg_;
// };
class Timer
{
    public:
    Timer() : start_(0), time_(0) {}

    void start()
    {
        start_ = cv::getTickCount();
    }

    void stop()
    {
        CV_Assert(start_ != 0);
        int64 end = cv::getTickCount();
        time_ += end - start_;
        start_ = 0;
    }

    double time()
    {
        double ret = time_ / cv::getTickFrequency();
        time_ = 0;
        return ret;
    }

    private:
    int64 start_, time_;
};

namespace linemod
{
///******************************************   two stucture *****************************************
///
struct Feature
{
    int x;
    int y;
    int label;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    Feature() : x(0), y(0), label(0) {}
    Feature(int x, int y, int label);
};
inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct Template
{
    int width;
    int height;
    int tl_x;
    int tl_y;
    int pyramid_level;
    std::vector<Feature> features;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
};

///****************************************************************************************************
void colormap(const cv::Mat& quantized, cv::Mat& dst);

///******************************************   two Class template  ***********************************
class QuantizedPyramid
{
    public:
    virtual ~QuantizedPyramid() {}

    virtual void quantize(cv::Mat& dst) const = 0;
    virtual bool extractTemplate(CV_OUT Template& templ) const = 0;
    virtual void pyrDown() = 0;

    public:
    /// Candidate feature with a score
    size_t num_features;
    struct Candidate
    {
        Candidate(int x, int y, int label, float score);
        /// Sort candidates with high score to the front
        bool operator<(const Candidate& rhs) const
        {
        return score > rhs.score;
        }
        Feature f;
        float score;
    };
    
    static bool selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                        std::vector<Feature>& features,
                                        size_t num_features, float distance);
};
inline QuantizedPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

class Modality
{
    public:
    virtual ~Modality() {}
    /**
     * \brief Form a quantized image pyramid from a source image.
     *
     * \param[in] src  The source image. Type depends on the modality.
     * \param[in] mask Optional mask. If not empty, unmasked pixels are set to zero
     *                 in quantized image and cannot be extracted as features.
     */
    cv::Ptr<QuantizedPyramid> process(const cv::Mat& src,
                        const cv::Mat& mask = cv::Mat()) const 
                        {
                            return processImpl(src, mask);
                        }
    virtual cv::Ptr<QuantizedPyramid> processImpl(const cv::Mat& src,
                        const cv::Mat& mask) const =0;

    virtual cv::String name() const = 0;

    virtual void read(const cv::FileNode& fn) = 0;
    virtual void write(cv::FileStorage& fs) const = 0;
    static cv::Ptr<Modality> create(const cv::String& modality_type);
    static cv::Ptr<Modality> create(const cv::FileNode& fn);

};
////*************************************************************************************************



class ColorGradientPyramid : public QuantizedPyramid
{
public:
    ColorGradientPyramid(const cv::Mat &src, const cv::Mat &mask,
                                             float weak_threshold, size_t num_features,
                                             float strong_threshold);

    virtual void quantize(cv::Mat &dst) const;

    virtual bool extractTemplate(Template &templ) const;

    virtual void pyrDown();

public:
    void update();
    /// Candidate feature with a score
    // struct Candidate
    // {
    //     Candidate(int x, int y, int label, float score);

    //     /// Sort candidates with high score to the front
    //     bool operator<(const Candidate &rhs) const
    //     {
    //         return score > rhs.score;
    //     }

    //     Feature f;
    //     float score;
    // };

    cv::Mat src;
    cv::Mat mask;

    int pyramid_level;
    cv::Mat angle;
    cv::Mat magnitude;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    // static bool selectScatteredFeatures(const std::vector<Candidate> &candidates,
    //                                                                         std::vector<Feature> &features,
    //                                                                         size_t num_features, float distance);
};
// inline ColorGradientPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

class ColorGradient : public Modality
{
public:
    ColorGradient();
    ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

    virtual cv::String name() const;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;

    virtual void read(const cv::FileNode &fn);
    virtual void write(cv::FileStorage &fs) const;

    virtual cv::Ptr<QuantizedPyramid> processImpl(const cv::Mat& src,
                        const cv::Mat& mask) const
    {
        return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features, strong_threshold);
    }
};

struct Match
{
    Match()
    {
    }

    Match(int x, int y, float angle, float scale, float similarity, const std::string &class_id, int template_id);

    /// Sort matches with high similarity to the front
    bool operator<(const Match &rhs) const
    {
        // Secondarily sort on template_id for the sake of duplicate removal
        if (similarity != rhs.similarity)
            return similarity > rhs.similarity;
        else
            return template_id < rhs.template_id;
    }

    bool operator==(const Match &rhs) const
    {
        return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
    }

    int x;
    int y;
    float angle;
    float scale;
    float similarity;
    std::string class_id;
    int template_id;
};
inline Match::Match(int _x, int _y, float _angle, float _scale, float _similarity, const std::string &_class_id, int _template_id) : x(_x), y(_y), angle(_angle), scale(_scale), similarity(_similarity), class_id(_class_id), template_id(_template_id){}


///*************************************************************          < DepthNormal* >
class DepthNormalPyramid : public QuantizedPyramid
{
    public:
        DepthNormalPyramid(const cv::Mat& src, const cv::Mat& mask,
                            int distance_threshold, int difference_threshold, size_t num_features,
                            int extract_threshold);

        virtual void quantize(cv::Mat& dst) const;

        virtual bool extractTemplate(Template& templ) const;

        virtual void pyrDown();

    protected:
    cv::Mat mask;

    int pyramid_level;
    cv::Mat normal;

    size_t num_features;
    int extract_threshold;
};

class DepthNormal : public Modality
{
    public:
    /**
     * \brief Default constructor. Uses reasonable default parameter values.
     */
    DepthNormal();

    /**
     * \brief Constructor.
     *
     * \param distance_threshold   Ignore pixels beyond this distance.
     * \param difference_threshold When computing normals, ignore contributions of pixels whose
     *                             depth difference with the central pixel is above this threshold.
     * \param num_features         How many features a template must contain.
     * \param extract_threshold    Consider as candidate feature only if there are no differing
     *                             orientations within a distance of extract_threshold.
     */
    DepthNormal(int distance_threshold, int difference_threshold, size_t num_features,
                int extract_threshold);

    virtual cv::String name() const;

    virtual void read(const cv::FileNode& fn);
    virtual void write(cv::FileStorage& fs) const;

    int distance_threshold;
    int difference_threshold;
    size_t num_features;
    int extract_threshold;

    virtual cv::Ptr<QuantizedPyramid> processImpl(const cv::Mat& src,
                        const cv::Mat& mask) const
    {
        return cv::makePtr<DepthNormalPyramid>(src, mask, distance_threshold, difference_threshold,
                                     num_features, extract_threshold);
    }
    
};


///*************************************************************          < Detector >
class Detector
{
    public:
        /**
             * \brief Empty constructor, initialize with read().
             */
        Detector();
        Detector(std::vector<int> T, std::string mode);
        Detector(int num_features, std::vector<int> T, std::string mode);
        Detector(const std::vector< cv::Ptr<Modality> >& modalities, const std::vector<int>& T_pyramid, std::string mode);

        std::string detect_mode;
        std::vector<Match> match(std::vector<cv::Mat>& sources, float threshold,
                                                        const std::vector<std::string> &class_ids = std::vector<std::string>(),
                                                        const cv::Mat masks = cv::Mat()) const;

        int addTemplate(const cv::Mat sources, const std::string &class_id,
                                        const cv::Mat &object_mask, float angle, float scale, int num_features = 0);
        int addTemplate(const std::vector<cv::Mat>& sources, const std::string& class_id,
                                        const cv::Mat& object_mask, float angle, float scale, int num_features = 0);

        int addSyntheticTemplate(const std::vector<Template>& templates, const std::string& class_id);

        // const cv::Ptr<ColorGradient> &getModalities() const { return CG_modality; }
        const std::vector< cv::Ptr<Modality> >& getModalities() const { return modalities; }

        int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

        int pyramidLevels() const { return pyramid_levels; }

        const std::vector<Template> &getTemplates(const std::string &class_id, int template_id) const;

        int numTemplates() const;
        int numTemplates(const std::string &class_id) const;
        int numClasses() const { return static_cast<int>(class_templates.size()); }

        std::vector<std::string> classIds() const;

        void read(const cv::FileNode &fn);
        void write(cv::FileStorage &fs) const;

        void readClass(const cv::FileNode &fn, const std::string &class_id_override = "");
        void writeClass(const std::string &class_id, cv::FileStorage &fs) const;

        void readClasses(const std::vector<std::string> &class_ids,
                                        const std::string &format = "templates_%s.yml.gz");
        void writeClasses(const std::string &format = "templates_%s.yml.gz") const;

    protected:
        cv::Ptr<ColorGradient> CG_modality;
        std::vector< cv::Ptr<Modality> > modalities;
        int pyramid_levels;
        std::vector<int> T_at_level;

        typedef pair< std::vector<float>, std::vector<Template> > TemplatePyramid;  /// 增加一维用来存angle 和scale
        typedef std::map< std::string, std::vector<TemplatePyramid> > TemplatesMap;
        TemplatesMap class_templates;

        typedef std::vector<cv::Mat> LinearMemories;
        // Indexed as [pyramid level][ColorGradient][quantized label]
        typedef std::vector< std::vector<LinearMemories> > LinearMemoryPyramid;   // 这是一个3维的Mat数组, 每一个元素是一个Mat
                                                                                
        void matchClass(const LinearMemoryPyramid &lm_pyramid,
                                        const std::vector<cv::Size> &sizes,
                                        float threshold, std::vector<Match> &matches,
                                        const std::string &class_id,
                                        const std::vector<TemplatePyramid> &template_pyramids) const;
};

    cv::Ptr< Detector > getDefaultLINEMOD( std::string mode);
} // namespace linemod

namespace shape_based_matching {
class shapeInfo{
public:
    cv::Mat src;
    cv::Mat mask;

    std::vector<float> angle_range;
    std::vector<float> scale_range;

    float angle_step = 15;
    float scale_step = 0.5;
    float eps = 0.00001f;

    class shape_and_info{
    public:
        cv::Mat src;
        cv::Mat mask;
        float angle;
        float scale;
        shape_and_info(cv::Mat src_, cv::Mat mask_, float angle_, float scale_){
            src = src_;
            mask = mask_;
            angle = angle_;
            scale = scale_;
        }
    };
    std::vector<shape_and_info> infos;

    shapeInfo(cv::Mat src, cv::Mat mask = cv::Mat()){
        this->src = src;
        if(mask.empty()){
            // make sure we have masks
            this->mask = cv::Mat(src.size(), CV_8UC1, {255});
        }else{
            this->mask = mask;
        }
    }

    static cv::Mat transform(cv::Mat src, float angle, float scale){
        cv::Mat dst;

        cv::Point center(src.cols/2, src.rows/2);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
        cv::warpAffine(src, dst, rot_mat, src.size());

        return dst;
    }
    static void save_infos(std::vector<shapeInfo::shape_and_info>& infos, cv::Mat src, cv::Mat mask, std::string path = "infos.yaml"){
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        fs << "src" << src;
        fs << "mask" << mask;
        fs << "infos"
           << "[";
        for (int i = 0; i < infos.size(); i++)
        {
            fs << "{";
            fs << "angle" << infos[i].angle;
            fs << "scale" << infos[i].scale;
            fs << "}";
        }
        fs << "]";
    }
    static std::vector< std::vector<float> > load_infos(cv::Mat& src, cv::Mat& mask, std::string path = "info.yaml"){
        cv::FileStorage fs(path, cv::FileStorage::READ);

        fs["src"] >> src;
        fs["mask"] >> mask;
        std::vector< std::vector<float> > infos;
        cv::FileNode infos_fn = fs["infos"];
        cv::FileNodeIterator it = infos_fn.begin(), it_end = infos_fn.end();
        for (int i = 0; it != it_end; ++it, i++)
        {
            std::vector<float> info;
            info.push_back(float((*it)["angle"]));
            info.push_back(float((*it)["scale"]));
            infos.push_back(info);
        }
        return infos;
    }

    void produce_infos(){
        assert(angle_range.size() <= 2);
        assert(scale_range.size() <= 2);
        assert(angle_step > eps*10);
        assert(scale_step > eps*10);

        // make sure range not empty
        if(angle_range.size() == 0){
            angle_range.push_back(0);
        }
        if(scale_range.size() == 0){
            scale_range.push_back(1);
        }

        if(angle_range.size() == 1 && scale_range.size() == 1){
            float angle = angle_range[0];
            float scale = scale_range[0];
            cv::Mat src_transformed = transform(src, angle, scale);
            cv::Mat mask_transformed = transform(mask, angle, scale);
            mask_transformed = mask_transformed > 0; //make sure it's a mask after transform
            infos.emplace_back(src_transformed, mask_transformed, angle, scale);

        }else if(angle_range.size() == 1 && scale_range.size() == 2){
            assert(scale_range[1] > scale_range[0]);
            float angle = angle_range[0];
            for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step){
                cv::Mat src_transformed = transform(src, angle, scale);
                cv::Mat mask_transformed = transform(mask, angle, scale);
                mask_transformed = mask_transformed > 0; //make sure it's a mask after transform
                infos.emplace_back(src_transformed, mask_transformed, angle, scale);
            }
        }else if(angle_range.size() == 2 && scale_range.size() == 1){
            assert(angle_range[1] > angle_range[0]);
            float scale = scale_range[0];
            for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step){
                cv::Mat src_transformed = transform(src, angle, scale);
                cv::Mat mask_transformed = transform(mask, angle, scale);
                mask_transformed = mask_transformed > 0; //make sure it's a mask after transform
                infos.emplace_back(src_transformed, mask_transformed, angle, scale);
            }
        }else if(angle_range.size() == 2 && scale_range.size() == 2){
            assert(scale_range[1] > scale_range[0]);
            assert(angle_range[1] > angle_range[0]);
            for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step){
                for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step){
                    cv::Mat src_transformed = transform(src, angle, scale);
                    cv::Mat mask_transformed = transform(mask, angle, scale);
                    mask_transformed = mask_transformed > 0; //make sure it's a mask after transform
                    infos.emplace_back(src_transformed, mask_transformed, angle, scale);
                }
            }
        }
    }
};

}



#endif
