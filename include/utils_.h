#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc_c.h> // cvFindContours
#include <opencv2/imgproc.hpp>
// #include <opencv2/rgbd.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iterator>
#include <set>
#include <cstdio>
#include <iostream>
#include "line2Dup.h"

// Function prototypes
using namespace std;
using namespace cv;

void subtractPlane(const cv::Mat& depth, cv::Mat& mask, std::vector<CvPoint>& chain, double f);

std::vector<CvPoint> maskFromTemplate(const std::vector<linemod::Template>& templates,
                                      int num_modalities, cv::Point offset, cv::Size size,
                                      cv::Mat& mask, cv::Mat& dst);

void templateConvexHull(const std::vector<linemod::Template>& templates,
                        int num_modalities, cv::Point offset, cv::Size size,
                        cv::Mat& dst);

void drawResponse(const std::vector<linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T);

void writeLinemod(const cv::Ptr<linemod::Detector>& detector, const std::string& filename);
cv::Ptr<linemod::Detector> readLinemod(const std::string& filename, std::string detect_mode);


cv::Mat displayQuantized(const cv::Mat& quantized);

class Mouse
{
  public:
    static void start(const std::string& a_img_name)
    {
        cv::setMouseCallback(a_img_name.c_str(), Mouse::cv_on_mouse, 0);
    }
    static int event(void)
    {
      int l_event = mouse_event;
      mouse_event = -1;
      return l_event;
    }
    static int x(void)
    {
      return mouse_x;
    }
    static int y(void)
    {
      return mouse_y;
    }

  private:
    static void cv_on_mouse(int a_event, int a_x, int a_y, int, void *)
    {
      mouse_event = a_event;
      mouse_x = a_x;
      mouse_y = a_y;
    }

  static int mouse_event;
  static int mouse_x;
  static int mouse_y;
};
// int Mouse::mouse_event;
// int Mouse::mouse_x;
// int Mouse::mouse_y;

static void help()
{
  printf("Usage: openni_demo [templates.yml]\n\n"
         "Place your object on a planar, featureless surface. With the mouse,\n"
         "frame it in the 'color' window and right click to learn a first template.\n"
         "Then press 'l' to enter online learning mode, and move the camera around.\n"
         "When the match score falls between 90-95%% the demo will add a new template.\n\n"
         "Keys:\n"
         "\t h   -- This help page\n"
         "\t l   -- Toggle online learning\n"
         "\t m   -- Toggle printing match result\n"
         "\t t   -- Toggle printing timings\n"
         "\t w   -- Write learned templates to disk\n"
         "\t [ ] -- Adjust matching threshold: '[' down,  ']' up\n"
         "\t q   -- Quit\n\n");
}

