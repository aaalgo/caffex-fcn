#include <iostream>
#include <fstream>
#include <algorithm>
#define CAFFEX_IMPL 1
#include "caffex.h"

namespace caffex {

Caffex::Caffex(string const& model_dir, unsigned batch)
    : net(model_dir + "/caffe.model", TEST)
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    BOOST_VERIFY(0);    // GPU is not supported here
#endif
    BOOST_VERIFY(batch >= 1);

    CHECK_EQ(net.num_inputs(), 1) << "Network should have exactly one input." << net.num_inputs();
    input_blob = net.input_blobs()[0];
    input_batch = batch;
    input_channels = input_blob->shape(1);

    CHECK(input_channels == 3 || input_channels == 1)
        << "Input layer should have 1 or 3 channels.";

    net.CopyTrainedLayersFrom(model_dir + "/caffe.params");
    // resize to required batch size
    input_blob->Reshape(input_batch, input_channels, 1, 1); // placeholder, not used anyway
    net.Reshape();

    // set mean file
    mean = 0;
    string mean_file = model_dir + "/caffe.mean";
    std::ifstream test(mean_file.c_str());
    if (test) {
        test >> mean;
    }
    {
        string blobs_file = model_dir + "/blobs";
        std::ifstream is(blobs_file.c_str());
        string blob;
        CHECK(is) << "cannot open blobs file.";
        while (is >> blob) {
            shared_ptr<Blob<float>> b = net.blob_by_name(blob);
            output_blobs.push_back(b);
        }
    }
}

void Caffex::wrapInputLayer (std::vector<cv::Mat>* channels) {
    int input_height = input_blob->shape(2);
    int input_width = input_blob->shape(3);
    float *input_data = input_blob->mutable_cpu_data();
    for (int i = 0; i < input_batch; ++i) {
        for (int j = 0; j < input_channels; ++j) {
            cv::Mat m(input_height, input_width, CV_32FC1, input_data);
            channels->push_back(m);
            input_data += m.total();
        }
    }
}

void Caffex::preprocess(cv::Mat const &img, cv::Mat *channels) {
    if (img.total() == 0) {
        for (int i = 0; i < input_channels; ++i) {
            channels[i].setTo(cv::Scalar(0));
        }
        return;
    }
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && input_channels == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && input_channels == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && input_channels == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && input_channels == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
    sample = img;

    cv::Mat sample_float;
    if (input_channels == 3)
    sample.convertTo(sample_float, CV_32FC3);
    else
    sample.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized = sample_float - mean;

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_normalized, channels);

}

void Caffex::extractOutputValues (float *ptr, int output_dim, int n) {
    for (auto const &blob: output_blobs) {
      int blob_dim = blob->count() / input_batch;
      float const *from_begin = blob->cpu_data() + blob->offset(0);
      float *to_begin = ptr;
      for (int i = 0; i < n; ++i) {
          float const *from_end = from_begin + blob_dim;
          std::copy(from_begin, from_end, to_begin);
          from_begin = from_end;
          to_begin += output_dim;
      }
      ptr += blob_dim;
    } 
}

void Caffex::checkReshape (cv::Mat const &image) {
    int input_height = input_blob->shape(2);
    int input_width = input_blob->shape(3);
    if ((input_width != image.cols)
            || (input_height != image.rows)) {
        input_blob->Reshape(input_batch, input_channels, image.rows, image.cols);
        net.Reshape();
    }
}

int Caffex::dim () const {
    int v = 0;
    for (auto const &b: output_blobs) {
        v += b->count() / input_batch;
    }
    return v;
}

void Caffex::apply (const cv::Mat &image, vector<float> *ft) {
    checkReshape(image);
    int output_dim = dim(); // output dim changed after reshape

    vector<cv::Mat> channels;
    wrapInputLayer(&channels);
    preprocess(image, &channels);
    CHECK(reinterpret_cast<float*>(channels[0].data) == net.input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
    net.ForwardPrefilled();
    ft->resize(output_dim);
    extractOutputValues(&ft->at(0), output_dim, 1);
}


void Caffex::apply(vector<cv::Mat> const &images, cv::Mat *ft) {
    CHECK(!images.empty()) << "must input >= 1 images";
    CHECK(images.size() <= input_batch) << "Too many input images.";
    for (unsigned i = 1; i < images.size(); ++i) {
        CHECK(images[i].size() == images[0].size()) << "all images must be the same size";
    }
    checkReshape(images[0]);
    int output_dim = dim(); // output dim changed after reshape

    vector<cv::Mat> channels;
    wrapInputLayer(&channels);
    preprocess(images, &channels);
    CHECK(reinterpret_cast<float*>(channels[0].data) == net.input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
    net.ForwardPrefilled();
    ft->create(images.size(), output_dim, CV_32FC1);
    extractOutputValues(ft->ptr<float>(0), output_dim, images.size());
}

}


