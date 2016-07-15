#define CPU_ONLY 1
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <boost/scoped_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <opencv2/opencv.hpp>
#include <cppformat/format.h>
#include <json11.hpp>
#include "glog/logging.h"

using namespace std;
using namespace boost;
using namespace json11;
namespace fs = boost::filesystem;

class Shape {
public:
    virtual void draw (cv::Mat *, cv::Scalar v, int thickness = CV_FILLED) const = 0;
    static std::shared_ptr<Shape> create (Json const &geo);
};

class Box: public Shape {
    cv::Rect_<float> rect;
public:
    Box (Json const &geo) {
        rect.x = geo["x"].number_value();
        rect.y = geo["y"].number_value();
        rect.width = geo["width"].number_value();
        rect.height = geo["height"].number_value();
    }
    virtual void draw (cv::Mat *m, cv::Scalar v, int thickness) const {
        cv::Rect box;
        box.x = std::round(m->cols * rect.x);
        box.y = std::round(m->rows * rect.y);
        box.width = std::round(m->cols * rect.width);
        box.height = std::round(m->rows * rect.height);
        cv::rectangle(*m, box, v, thickness);
    }
};

class Poly: public Shape {
    vector<cv::Point2f> points;
public:
    Poly (Json const &geo) {
        for (auto const &p: geo["points"].array_items()) {
            points.emplace_back(p["x"].number_value(), p["y"].number_value());
        }
    }
    virtual void draw (cv::Mat *m, cv::Scalar v, int thickness) const {
        vector<cv::Point> ps(points.size());
        for (unsigned i = 0; i < ps.size(); ++i) {
            auto const &from = points[i];
            auto &to = ps[i];
            to.x = std::round(from.x * m->cols);
            to.y = std::round(from.y * m->rows);
        }
        cv::Point const *pps = &ps[0];
        int const nps = ps.size();
        if (thickness == CV_FILLED) {
            cv::fillPoly(*m, &pps, &nps, 1, v);
        }
        else {
            cv::polylines(*m, &pps, &nps, 1, true, v, thickness);
        }
    }
};

std::shared_ptr<Shape> Shape::create (Json const &geo) {
    string type = geo["type"].string_value();
    if (type == "rect") {
        return std::shared_ptr<Shape>(new Box(geo["geometry"]));
    }
    else if (type == "polygon") {
        return std::shared_ptr<Shape>(new Poly(geo["geometry"]));
    }
}

class Annotation {
    vector<std::shared_ptr<Shape>> shapes;
public:
    Annotation () {}
    Annotation (string const &txt) {
        string err;
        Json json = Json::parse(txt, err);
        if (err.size()) {
            LOG(ERROR) << "Bad json: " << err << " (" << txt << ")";
            return;
        }
        for (auto const &x: json["shapes"].array_items()) {
            shapes.emplace_back(Shape::create(x));
        }
    }

    void draw (cv::Mat *m, cv::Scalar v, int thickness = -1) const {
        for (auto const &p: shapes) {
            p->draw(m, v, thickness);
        }
    }
};


struct Sample {
    string url;
    Annotation anno;
};

bool IsURL (std::string const &url) {
    if (url.compare(0, 7, "http://") == 0) return true;
    if (url.compare(0, 8, "https://") == 0) return true;
    if (url.compare(0, 6, "ftp://") == 0) return true;
    return false;
}   

string download_agent;
int download_timeout = 5;
fs::path temp_dir("/tmp");

fs::path temp_path (const fs::path& model="%%%%-%%%%-%%%%-%%%%") {
    return fs::unique_path(temp_dir / model);
}

void Download (const std::string &url, fs::path const &path) {
    int timeout = download_timeout;
    if (timeout <= 0) {
        timeout = 5;
    }
    ostringstream ss;
    ss << "wget --output-document=" << path.native() << " --tries=1 -nv --no-check-certificate";
    ss << " --quiet --timeout=" << timeout;
    if (download_agent.size()) {
        ss << " --user-agent=" << download_agent;
    }
    ss << ' ' << '"' << url << '"';
    string cmd = ss.str();
    LOG(INFO) << cmd;
    ::system(cmd.c_str()) == 0;
}

cv::Mat imreadx (string const &url) {
    cv::Mat v;
    if (IsURL(url)) {
        fs::path path = temp_path();
        Download(url, path);
        v = cv::imread(path.native(), -1);
        if (!v.data) {
            LOG(ERROR) << "Failed to download " << url;
        }
        fs::remove(path);
    }
    else {
        v = cv::imread(url, -1);
    }

    if (!v.data) return v;
    // TODO! support color image
    return v;
}

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string output_path;
    int thickness;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("output,o", po::value(&output_path), "")
    ("timeout", po::value(&download_timeout)->default_value(5), "")
    ("agent", po::value(&download_agent), "")
    ("log-level,v", po::value(&FLAGS_minloglevel)->default_value(1), "")
    ("tmp", po::value(&temp_dir), "")
    ("thickness,t", po::value(&thickness)->default_value(3), "")
    ;

    po::positional_options_description p;
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || output_path.empty()) {
        cerr << desc;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    Sample s;
    string line;
    while (getline(cin, line)) {
        vector<string> ss;
        split(ss, line, is_any_of("\t"), token_compress_off);
        if (ss.size() != 2) {
            cerr << "Bad line: " << line << endl;
            continue;
        }
        s.url = ss[0];
        s.anno = Annotation(ss[1]);
        /*
        if (!fs::is_regular_file(fs::path(root_dir + path))) {
            LOG(ERROR) << "Cannot find regular file: " << path;
            continue;
        }
        */
        cv::Mat image = imreadx(s.url);
        if (image.channels() == 1) {
            s.anno.draw(&image, cv::Scalar(0), 1);
        }
        else {
            s.anno.draw(&image, cv::Scalar(0, 0, 0xFF), thickness);
        }
        cv::imwrite(output_path, image);
        break;
    }

    return 0;
}

