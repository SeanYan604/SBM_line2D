#include "../include/utils_.h"
#include "../include/line2Dup.h"

// Functions to store detector and templates in single XML/YAML file
cv::Ptr<linemod::Detector> readLinemod(const std::string& filename, std::string detect_mode)
{
  cout << "begin read temp !" << endl;
  cv::Ptr<linemod::Detector> detector = linemod::getDefaultLINEMOD(detect_mode);
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector->read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
  {
    cout << "begin read class !" << endl;
    detector->readClass(*i);
  }

  return detector;
}

void writeLinemod(const cv::Ptr<linemod::Detector>& detector, const std::string& filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector->write(fs);

  std::vector<std::string> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int)ids.size(); ++i)
  {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}"; // current class
  }
  fs << "]"; // classes
}


