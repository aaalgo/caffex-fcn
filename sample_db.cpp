#define CPU_ONLY 1
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <boost/scoped_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <opencv2/opencv.hpp>
#include <cppformat/format.h>

#include <glog/logging.h>

#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>

using namespace std;
using namespace boost;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
namespace fs = boost::filesystem;

string backend("lmdb");

cv::Mat DatumToCVMat (Datum &datum) {
    int c = datum.channels();
    int w = datum.width();
    int h = datum.height();

    CHECK((c == 1) || (c == 3));
    cv::Mat im;
    if (c == 1) {
        im = Mat(h, w, CV_8U, const_cast<char *>(&datum.data()[0])).clone();
    }
    else {
        int sz = w * h;
        cv::Mat chs[] = {cv::Mat(h, w, CV_8U, const_cast<char *>(&datum.data()[0])),
                         cv::Mat(h, w, CV_8U, const_cast<char *>(&datum.data()[sz])),
                         cv::Mat(h, w, CV_8U, const_cast<char *>(&datum.data()[2*sz]))};
        cv::merge(chs, 3, im);
    }
    return im;
}

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string image_db_dir;
    string label_db_dir;
    string output_dir;
    int N, M;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("image", po::value(&image_db_dir), "")
    ("label", po::value(&label_db_dir), "")
    ("output", po::value(&output_dir), "")
    (",N", po::value(&N)->default_value(200), "")
    (",M", po::value(&M)->default_value(10000), "")
    ;

    po::positional_options_description p;
    p.add("image", 1);
    p.add("label", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || image_db_dir.empty() || label_db_dir.empty() || output_dir.empty()) {
        cerr << desc;
        return 1;
    }
  // Create new DB
    scoped_ptr<db::DB> image_db(db::GetDB(backend));
    image_db->Open(image_db_dir, db::READ);
    scoped_ptr<db::Cursor> image_cur(image_db->NewCursor());
    image_cur->SeekToFirst();

    scoped_ptr<db::DB> label_db(db::GetDB(backend));
    label_db->Open(label_db_dir, db::READ);
    scoped_ptr<db::Cursor> label_cur(label_db->NewCursor());
    label_cur->SeekToFirst();

    boost::progress_display progress(M, cerr);
    vector<cv::Mat> samples;
    samples.reserve(N);
    for (unsigned m = 0; m < M; ++m) {
        cv::Mat *target = nullptr;
        if (samples.size() < N) {
            samples.emplace_back();
            target = &samples.back();
        }
        else {
            CHECK(m >= samples.size());
            int R = rand() % m;
            if (R < samples.size()) {
                target = &samples[R];
            }
        }
        if (target) {
            string iv = image_cur->value();
            string lv = label_cur->value();
            CHECK(image_cur->key() == label_cur->key());
            Datum datum;
            bool r = datum.ParseFromString(iv);
            CHECK(r);
            cv::Mat im = DatumToCVMat(datum);
            r = datum.ParseFromString(lv);
            CHECK(r);
            cv::Mat lm = DatumToCVMat(datum);
            if (im.channels() == 3) {
                cvtColor(lm, lm, CV_GRAY2BGR);
            }
            lm *= 255;
            lm += im;
            cv::hconcat(im, lm, *target);
        }
        image_cur->Next();
        label_cur->Next();
        if (!image_cur->valid()) {
            CHECK(!label_cur->valid());
            image_cur->SeekToFirst();
            label_cur->SeekToFirst();
        }
        ++progress;
    }
    fs::path root(output_dir);
    fs::create_directories(root);
    for (unsigned i = 0; i < samples.size(); ++i) {
        fs::path out(root/fs::path(fmt::format("{}.png", i)));
        cv::imwrite(out.native(), samples[i]);
    }

    return 0;
}

