#include <boost/assert.hpp>
#include <iostream>
#include <boost/program_options.hpp>
#include "caffex.h"


using namespace std;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string model;
    string ipath;
    string opath;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("model", po::value(&model)->default_value("model"), "")
    ("input", po::value(&ipath), "")
    ("output", po::value(&opath), "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || ipath.empty()) {
        cerr << desc;
        return 1;
    }

    caffex::Caffex det(model);
    cv::Mat ret;
    cv::Mat input = cv::imread(ipath, CV_LOAD_IMAGE_COLOR);
    BOOST_VERIFY(input.data);
    vector<float> resp;
    det.apply(input, &resp);
    for (auto &v: resp) {
        v = 1.0 - v;
    }
    cv::Mat fl;
    input.convertTo(fl, CV_32FC3);
    cv::Mat prob(input.size(), CV_32F, &resp[0]);
    vector<cv::Mat> chs{prob, prob, prob};
    cv::Mat prob3d;
    cv::merge(&chs[0], 3, prob3d);
    cv::Mat mask = fl.mul(prob3d);
    prob3d *= 255;
    cv::hconcat(mask, prob3d, mask);
    cv::hconcat(mask, fl, fl);
    cv::imwrite(opath, fl);
    return 0;
}

