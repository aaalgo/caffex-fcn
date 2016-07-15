#include <iostream>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "caffex.h"
#include "bbox.h"

static float LimitSize (cv::Mat input, int max_size, cv::Mat *output) {
    if (input.rows == 0) {
        *output = cv::Mat();
        return 0;
    }
    float scale = 1.0;
    int maxs = std::max(input.cols, input.rows);
    // large side > max
    if ((max_size > 0) && (maxs > max_size)) {
        cv::Mat tmp;
        scale = 1.0 * maxs / max_size;
        cv::resize(input, tmp, cv::Size(input.cols * max_size / maxs, input.rows * max_size / maxs));            input = tmp;
    }
    *output = input;
    return scale; 
}


using namespace std;
using namespace boost;
namespace fs = boost::filesystem;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string model;
    vector<fs::path> ipaths;
    fs::path odir;
    int max;
    float b_th;
    float b_keep;
    float b_sth;
    float b_ssp;


    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("model", po::value(&model)->default_value("model"), "")
    ("input", po::value(&ipaths), "")
    ("output,o", po::value(&odir)->default_value("vis"), "")
    ("th", po::value(&b_th)->default_value(0.05), "")
    ("keep", po::value(&b_keep)->default_value(0.95), "")
    ("sth", po::value(&b_sth)->default_value(0.2), "")
    ("ssp", po::value(&b_ssp)->default_value(0.2), "")
    ("max", po::value(&max)->default_value(-1), "")
    ;


    po::positional_options_description p;
    p.add("input", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help")) {
        cerr << desc;
        return 1;
    }

    if (ipaths.empty()) {
        string line;
        while (cin >> line) {
            ipaths.emplace_back(line);
        }
    }

    caffex::Caffex det(model);
    BBoxDetector bdet(b_th, b_keep, b_sth);

    unsigned cnt = 0;
    fs::create_directories(odir);
    for (auto const &path: ipaths) {
        cv::Mat ret;
        cv::Mat input = cv::imread(path.native(), CV_LOAD_IMAGE_COLOR);
        BOOST_VERIFY(input.data);
        if (max > 0) {
            cv::Mat tmp;
            LimitSize(input, max, &tmp);
            input = tmp;
        }
        vector<float> resp;
        det.apply(input, &resp);
        for (auto &v: resp) {
            v = 1.0 - v;
        }
        cv::Mat fl;
        input.convertTo(fl, CV_32FC3);
        cv::Mat prob(input.size(), CV_32F, &resp[0]);
        vector<BBox> boxes;
        bdet.apply(prob, &boxes);

        vector<cv::Mat> chs{prob, prob, prob};
        cv::Mat prob3d;
        cv::merge(&chs[0], 3, prob3d);
        cv::Mat mask = fl.mul(prob3d);
        prob3d *= 255;
        float best = 0;
        for (auto const &box: boxes) {
            if (box.score > best) best = box.score;
            cv::rectangle(prob3d, box.box, cv::Scalar(0, 0, 0xFF), 2);
            cv::rectangle(mask, box.box, cv::Scalar(0, 0, 0xFF), 2);
            cv::rectangle(fl, box.box, cv::Scalar(0, 0, 0xFF), 2);
        }
        cout << path << '\t' << best << endl;
        cv::hconcat(mask, prob3d, mask);
        cv::hconcat(mask, fl, fl);
        cv::imwrite((odir/(lexical_cast<string>(cnt++)+".jpg")).native(), fl);
    }
    return 0;
}

