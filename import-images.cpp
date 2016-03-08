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
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <opencv2/opencv.hpp>
#include <cppformat/format.h>
#include <json11.hpp>
#include "glog/logging.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace std;
using namespace boost;
using namespace json11;
using namespace caffe;  // NOLINT(build/namespaces)
namespace fs = boost::filesystem;

string backend("lmdb");

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
    int id;
    string url;
    Annotation anno;
};

bool gray = false;

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

bool Download (const std::string &url, fs::path const &path) {
    int timeout = download_timeout;
    if (timeout <= 0) {
        timeout = 5;
    }
    ostringstream ss;
    ss << "wget --output-document=" << path.native() << " --tries=1 --nv --no-check-certificate";
    ss << " --quiet --timeout=" << timeout;
    if (download_agent.size()) {
        ss << " --user-agent=" << download_agent;
    }
    string cmd = ss.str();
    return ::system(cmd.c_str()) == 0;
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
    if (gray) {
        if (v.channels() == 3) {
            cv::cvtColor(v, v, CV_BGR2GRAY);
        }
        else CHECK(v.channels() == 1);
    }
    else {
        if (v.channels() == 1) {
            cv::cvtColor(v, v, CV_GRAY2BGR);
        }
        else CHECK(v.channels() == 3);
    }
    // always to gray
    return v;
}

float sampler_angle = 10 * M_PI / 180;
float sampler_scale = 0.25;
class Sampler {
    // regular
    /*
    float max_color;
    std::uniform_real_distribution<float> delta_color; //(min_R, max_R);
    */
    std::uniform_real_distribution<float> linear_angle;
    std::uniform_real_distribution<float> linear_scale;
    std::default_random_engine e;
public:
    Sampler ()
        : /* max_color(config.get<float>("adsb2.aug.color", 20)),
        delta_color(-max_color, max_color), */
        linear_angle(-sampler_angle, sampler_angle),
        linear_scale(-sampler_scale, sampler_scale)
    {
    }

    bool linear (cv::Mat from_image,
                cv::Mat from_label,
                cv::Mat *to_image,
                cv::Mat *to_label, bool no_perturb = false) {
        if (no_perturb) {
            *to_image = from_image;
            *to_label = from_label;
            return true;
        }
        //float color, angle, scale, flip = false;
        float angle, scale, flip = false;
#pragma omp critical
        {
            //color = delta_color(e);
            angle = linear_angle(e);
            scale = std::exp(linear_scale(e));
            //flip = ((e() % 2) == 1);
        }
        cv::Mat image, label;
        if (flip) {
            cv::flip(from_image, image, 1);
            cv::flip(from_label, label, 1);
        }
        else {
            image = from_image;
            label = from_label;
        }
        cv::Mat rot = cv::getRotationMatrix2D(cv::Point(image.cols/2, image.rows/2), angle, scale);
        cv::warpAffine(image, *to_image, rot, image.size());
        cv::warpAffine(label, *to_label, rot, label.size(), cv::INTER_NEAREST); // cannot interpolate labels
        /*
        *to_image += color;
        */
        return true;
    }
};

int replicate = 1;
void import (vector<Sample> const &samples, fs::path const &dir, bool test_set = false) {
    CHECK(fs::create_directories(dir));
    fs::path image_path = dir / fs::path("images");
    fs::path label_path = dir / fs::path("labels");
    fs::path sample_path = dir / fs::path("samples");
  // Create new DB
    scoped_ptr<db::DB> image_db(db::GetDB(backend));
    image_db->Open(image_path.string(), db::NEW);
    scoped_ptr<db::Transaction> image_txn(image_db->NewTransaction());

    scoped_ptr<db::DB> label_db(db::GetDB(backend));
    label_db->Open(label_path.string(), db::NEW);
    scoped_ptr<db::Transaction> label_txn(label_db->NewTransaction());

    Sampler sampler;
    int count = 0;
    vector<unsigned> index(samples.size());
    for (unsigned i = 0; i < index.size(); ++i) {
        index[i] = i;
    }
    int n_rep = test_set ? 1 : replicate;
    for (int rep = 0; rep < n_rep; ++rep) {
        if (rep > 0) {
            random_shuffle(index.begin(), index.end());
        }
        for (unsigned sid: index) {
            auto const &sample = samples[sid];
            Datum datum;
            string key = lexical_cast<string>(sample.id), value;
            cv::Mat raw_image = imreadx(sample.url);
            cv::Mat raw_label(raw_image.size(), CV_8UC1, 0);
            sample.anno.draw(&raw_label, cv::Scalar(1));
            cv::Mat image, label;
            sampler.linear(raw_image, raw_label, &image, &label, rep == 0);

            caffe::CVMatToDatum(image, &datum);
            datum.set_label(0);
            CHECK(datum.SerializeToString(&value));
            image_txn->Put(key, value);

            caffe::CVMatToDatum(label, &datum);
            datum.set_label(0);
            CHECK(datum.SerializeToString(&value));
            label_txn->Put(key, value);

            if (++count % 1000 == 0) {
                // Commit db
                image_txn->Commit();
                image_txn.reset(image_db->NewTransaction());
                label_txn->Commit();
                label_txn.reset(label_db->NewTransaction());
            }
        }
    }
    if (count % 1000 != 0) {
      image_txn->Commit();
      label_txn->Commit();
    }
}

void save_list (vector<Sample> const &samples, fs::path path) {
    fs::ofstream os(path);
    for (auto const &s: samples) {
        os << s.url << endl;
    }
}

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string list_path;
    string root_dir;
    string output_dir;
    bool full = false;
    int F;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("list", po::value(&list_path), "")
    ("root", po::value(&root_dir)->default_value(""), "make sure dir ends with /")
    ("fold,f", po::value(&F)->default_value(1), "")
    ("full", "")
    ("gray", "")
    ("output,o", po::value(&output_dir), "")
    ("timeout", po::value(&download_timeout)->default_value(5), "")
    ("agent", po::value(&download_agent), "")
    ;

    po::positional_options_description p;
    p.add("list", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || list_path.empty() || output_dir.empty()) {
        cerr << desc;
        return 1;
    }
    CHECK(F >= 1);
    full = vm.count("full") > 0;
    if (vm.count("gray")) gray = true;

    vector<Sample> samples;
    {
        Sample s;
        ifstream is(list_path.c_str());
        s.id = 0;
        string line;
        while (getline(is, line)) {
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
            samples.push_back(s);
            ++s.id;
        }
        LOG(INFO) << "Loaded " << samples.size() << " samples." << endl;
    }

    if (F == 1) {
        import(samples, fs::path(output_dir));
        return 0;
    }
    // N-fold cross validation
    vector<vector<Sample>> folds(F);
    random_shuffle(samples.begin(), samples.end());
    for (unsigned i = 0; i < samples.size(); ++i) {
        folds[i % F].push_back(samples[i]);
    }

    for (unsigned f = 0; f < F; ++f) {
        vector<Sample> const &val = folds[f];
        // collect training examples
        vector<Sample> train;
        for (unsigned i = 0; i < F; ++i) {
            if (i == f) continue;
            train.insert(train.end(), folds[i].begin(), folds[i].end());
        }
        fs::path fold_path(output_dir);
        if (full) {
            fold_path /= lexical_cast<string>(f);
        }
        CHECK(fs::create_directories(fold_path));
        save_list(train, fold_path / fs::path("train.list"));
        save_list(val, fold_path / fs::path("val.list"));
        import(train, fold_path / fs::path("train"));
        import(val, fold_path / fs::path("val"), true);
        if (!full) break;
    }

    return 0;
}

