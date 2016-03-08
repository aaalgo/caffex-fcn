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

fs::path cache_dir;
void Checksum (void const *data, unsigned length, std::string *checksum);

fs::path cache_path (const std::string &url) {
    string sum;
    Checksum(&url[0], url.size(), &sum);
    return cache_dir / fs::path(sum);
}

fs::path Download (const std::string &url) {
    fs::path path = cache_path(url);
    if (fs::exists(path)) {
        return path;
    }
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
    return path;
}

int max_size = 600;

void LimitSize (cv::Mat input, cv::Mat *output) {
    if (input.rows == 0) {
        *output = cv::Mat();
        return;
    }
    float scale = 1.0;
    int maxs = std::max(input.cols, input.rows);
    // large side > max
    if (maxs > max_size) {
        cv::Mat tmp;
        scale = 1.0 * maxs / max_size;
        cv::resize(input, tmp, cv::Size(input.cols * max_size / maxs, input.rows * max_size / maxs));
        input = tmp;
    }
    *output = input;
}

cv::Mat imreadx (string const &url) {
    cv::Mat v;
    if (IsURL(url)) {
        fs::path path = Download(url);
        v = cv::imread(path.native(), -1);
        if (!v.data) {
            LOG(ERROR) << "Failed to download " << url;
        }
        //fs::remove(path);
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
    LimitSize(v, &v);
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
            string key = lexical_cast<string>(count), value;
            cv::Mat raw_image = imreadx(sample.url);
            if (!raw_image.data) {
                LOG(ERROR) << "fail to load url: " << sample.url;
                continue;
            }
            cv::Mat raw_label(raw_image.size(), CV_8UC1, cv::Scalar(0));
            sample.anno.draw(&raw_label, cv::Scalar(1));
            cv::Mat image, label;
            //sampler.linear(raw_image, raw_label, &image, &label, rep == 0);
            image = raw_image;
            label = raw_label;

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
    ("replicate,R", po::value(&replicate)->default_value(1), "")
    ("sangle", po::value(&sampler_angle)->default_value(sampler_angle), "")
    ("sscale", po::value(&sampler_scale)->default_value(sampler_scale), "")
    ("max", po::value(&max_size)->default_value(max_size), "")
    ("log-level,v", po::value(&FLAGS_minloglevel)->default_value(1), "")
    ("cache", po::value(&cache_dir)->default_value(".caffex_cache"), "")
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

    google::InitGoogleLogging(argv[0]);
    fs::create_directories(cache_dir);
    vector<Sample> samples;
    {
        Sample s;
        ifstream is(list_path.c_str());
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

namespace from_boost_uuid_detail {

BOOST_STATIC_ASSERT(sizeof(unsigned char)*8 == 8);
BOOST_STATIC_ASSERT(sizeof(unsigned int)*8 == 32);

inline unsigned int left_rotate(unsigned int x, std::size_t n)
{
    return (x<<n) ^ (x>> (32-n));
}

class sha1
{
public:
    typedef unsigned int(&digest_type)[5];
public:
    sha1();

    void reset();

    void process_byte(unsigned char byte);
    void process_block(void const* bytes_begin, void const* bytes_end);
    void process_bytes(void const* buffer, std::size_t byte_count);

    void get_digest(digest_type digest);

private:
    void process_block();
    void process_byte_impl(unsigned char byte);

private:
    unsigned int h_[5];

    unsigned char block_[64];

    std::size_t block_byte_index_;
    std::size_t bit_count_low;
    std::size_t bit_count_high;
};

inline sha1::sha1()
{
    reset();
}

inline void sha1::reset()
{
    h_[0] = 0x67452301;
    h_[1] = 0xEFCDAB89;
    h_[2] = 0x98BADCFE;
    h_[3] = 0x10325476;
    h_[4] = 0xC3D2E1F0;

    block_byte_index_ = 0;
    bit_count_low = 0;
    bit_count_high = 0;
}

inline void sha1::process_byte(unsigned char byte)
{
    process_byte_impl(byte);

    if (bit_count_low < 0xFFFFFFF8) {
        bit_count_low += 8;
    } else {
        bit_count_low = 0;

        if (bit_count_high <= 0xFFFFFFFE) {
            ++bit_count_high;
        } else {
            BOOST_THROW_EXCEPTION(std::runtime_error("sha1 too many bytes"));
        }
    }
}

inline void sha1::process_byte_impl(unsigned char byte)
{
    block_[block_byte_index_++] = byte;

    if (block_byte_index_ == 64) {
        block_byte_index_ = 0;
        process_block();
    }
}

inline void sha1::process_block(void const* bytes_begin, void const* bytes_end)
{
    unsigned char const* begin = static_cast<unsigned char const*>(bytes_begin);
    unsigned char const* end = static_cast<unsigned char const*>(bytes_end);
    for(; begin != end; ++begin) {
        process_byte(*begin);
    }
}

inline void sha1::process_bytes(void const* buffer, std::size_t byte_count)
{
    unsigned char const* b = static_cast<unsigned char const*>(buffer);
    process_block(b, b+byte_count);
}

inline void sha1::process_block()
{
    unsigned int w[80];
    for (std::size_t i=0; i<16; ++i) {
        w[i]  = (block_[i*4 + 0] << 24);
        w[i] |= (block_[i*4 + 1] << 16);
        w[i] |= (block_[i*4 + 2] << 8);
        w[i] |= (block_[i*4 + 3]);
    }
    for (std::size_t i=16; i<80; ++i) {
        w[i] = left_rotate((w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]), 1);
    }

    unsigned int a = h_[0];
    unsigned int b = h_[1];
    unsigned int c = h_[2];
    unsigned int d = h_[3];
    unsigned int e = h_[4];

    for (std::size_t i=0; i<80; ++i) {
        unsigned int f;
        unsigned int k;

        if (i<20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999;
        } else if (i<40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } else if (i<60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }

        unsigned temp = left_rotate(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = left_rotate(b, 30);
        b = a;
        a = temp;
    }

    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
}

inline void sha1::get_digest(digest_type digest)
{
    // append the bit '1' to the message
    process_byte_impl(0x80);

    // append k bits '0', where k is the minimum number >= 0
    // such that the resulting message length is congruent to 56 (mod 64)
    // check if there is enough space for padding and bit_count
    if (block_byte_index_ > 56) {
        // finish this block
        while (block_byte_index_ != 0) {
            process_byte_impl(0);
        }

        // one more block
        while (block_byte_index_ < 56) {
            process_byte_impl(0);
        }
    } else {
        while (block_byte_index_ < 56) {
            process_byte_impl(0);
        }
    }

    // append length of message (before pre-processing) 
    // as a 64-bit big-endian integer
    process_byte_impl( static_cast<unsigned char>((bit_count_high>>24) & 0xFF) );
    process_byte_impl( static_cast<unsigned char>((bit_count_high>>16) & 0xFF) );
    process_byte_impl( static_cast<unsigned char>((bit_count_high>>8 ) & 0xFF) );
    process_byte_impl( static_cast<unsigned char>((bit_count_high)     & 0xFF) );
    process_byte_impl( static_cast<unsigned char>((bit_count_low>>24) & 0xFF) );
    process_byte_impl( static_cast<unsigned char>((bit_count_low>>16) & 0xFF) );
    process_byte_impl( static_cast<unsigned char>((bit_count_low>>8 ) & 0xFF) );
    process_byte_impl( static_cast<unsigned char>((bit_count_low)     & 0xFF) );

    // get final digest
    digest[0] = h_[0];
    digest[1] = h_[1];
    digest[2] = h_[2];
    digest[3] = h_[3];
    digest[4] = h_[4];
}
}
void Checksum (void const *data, unsigned length, std::string *checksum) {
        uint32_t digest[5];
        from_boost_uuid_detail::sha1 sha1;
        sha1.process_block(data, data+length);
        sha1.get_digest(digest);
        static char const digits[] = "0123456789abcdef";
        checksum->clear();
        for(uint32_t c: digest) {
            checksum->push_back(digits[(c >> 28) & 0xF]);
            checksum->push_back(digits[(c >> 24) & 0xF]);
            checksum->push_back(digits[(c >> 20) & 0xF]);
            checksum->push_back(digits[(c >> 16) & 0xF]);
            checksum->push_back(digits[(c >> 12) & 0xF]);
            checksum->push_back(digits[(c >> 8) & 0xF]);
            checksum->push_back(digits[(c >> 4) & 0xF]);
            checksum->push_back(digits[c & 0xF]);
        }
    }
