#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define DEVICE 1
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.6

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


static const int INPUT_H=640;
static const int INPUT_W=640;
static const int NUM_CLASSES = 2;
const char* INPUT_BLOB_NAME = "input0";
const char* OUTPUT_BLOB_NAME = "output0";
const std::vector<std::vector<float>> MIN_SIZES = {{16, 32}, {64, 128}, {256, 512}};
const std::vector<int> STEPS = {8,16,32};
const std::vector<float> VARIANCE = {0.1,0.2};
const std::vector<float> BGR = {104,117,123};
static Logger gLogger;

using namespace nvinfer1;



struct Object
{
    cv::Rect_<float> rect;
    float prob;
    std::vector<float> landmarks;
};



static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

float* blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c] - BGR[c];
            }
        }
    }
    return blob;
}

void PriorBox(std::vector<std::vector<float>>& priors)
{
    int step_num = 0;
    for (auto stride : STEPS)
    {
        int num_grid_y = INPUT_H / stride;
        int num_grid_x = INPUT_W / stride;
        for (int j=0;j<num_grid_y;j++)
        {
            for (int i=0;i<num_grid_x;i++)
            {
                for (int anchor_size:MIN_SIZES[step_num])
                {
                    float s_kx = float(anchor_size)/INPUT_W;
                    float s_ky = float(anchor_size)/INPUT_H;
                    float cx = (i + 0.5f) * stride / INPUT_W;
                    float cy = (j + 0.5f) * stride / INPUT_H;
                    priors.push_back({cx,cy,s_kx,s_ky});
                }
            }
        }
        step_num += 1;
    }

}

void GetProposal(float* prob,std::vector<Object>& objs,std::vector<std::vector<float>>& priors,float scale)
{
    const int num_priors = priors.size();
    const int single_obj_len = 16;
    for (int i=0;i<num_priors;i++)
    {
        float face_score = prob[i*single_obj_len + 5];
        if (face_score > BBOX_CONF_THRESH)
        {
            Object obj;
            auto prior = priors[i];
            float cx =  prob[i*single_obj_len+0] * VARIANCE[0] * prior[2] + prior[0];
            float cy =  prob[i*single_obj_len+1] * VARIANCE[0] * prior[3] + prior[1];
            float w = exp(prob[i*single_obj_len+2] * VARIANCE[1]) * prior[2];
            float h = exp(prob[i*single_obj_len+3] * VARIANCE[1]) * prior[3];
            float x = cx - w/2;
            float y = cy - h/2;
            obj.rect.x = x * INPUT_W / scale;
            obj.rect.y = y * INPUT_H / scale;
            obj.rect.width = w * INPUT_W / scale;
            obj.rect.height = h * INPUT_H / scale;
            obj.prob = face_score;

            for (int j=0;j<5;j++)
            {
                for (int d=0;d<2;d++)
                {
                    float point = prob[i*single_obj_len+6+j*2+d] * VARIANCE[0] * prior[2+d] + prior[d];   
                    if (d==0) 
                    {
                        obj.landmarks.push_back(point * INPUT_W / scale);
                    }
                    else
                    {
                        obj.landmarks.push_back(point * INPUT_H / scale);
                    }
                        
                }
                
            }
            objs.push_back(obj);
        }
    }
}

std::vector<int> decode(float* prob,std::vector<Object>& objs,float scale)
{
    std::vector<std::vector<float>> priors;
    PriorBox(priors);
    GetProposal(prob,objs,priors,scale);
    std::cout << objs.size()<<std::endl;
    
    qsort_descent_inplace(objs);

    std::vector<int> picked;

    nms_sorted_bboxes(objs,picked,NMS_THRESH);

    std::cout << picked.size() << std::endl;
    return picked;

}

void draw(cv::Mat& img,std::vector<Object>& objs,std::vector<int>& picked)
{
    cv::Mat image = img.clone();
    cv::Scalar color = cv::Scalar(255,0,25);
    for (int i : picked)
    {
        auto obj = objs[i];
        cv::rectangle(image,obj.rect,color,2);
    }
    cv::imwrite("res.jpg",image);
}

void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc,char** argv)
{
    cudaSetDevice(DEVICE);
    char* trtModelStream{nullptr};
    size_t size{0};

    if (argc==4 && std::string(argv[2])=="-i")
    {
        const std::string engine_file_path = argv[1];
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else
    {
        std::cout << "error input" << std::endl;
    }

    const std::string input_image_path = argv[3];

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    for(int j=1;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }

    static float* prob = new float[output_size];

    cv::Mat img = cv::imread(input_image_path);
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img);

    float* blob;
    blob = blobFromImage(pr_img);
    float scale = std::min(INPUT_W/(img_w*1.0),INPUT_H/(img_h*1.0));

    // run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, blob, prob, output_size, pr_img.size());
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<Object> objs;
    std::vector<int> picked = decode(prob,objs,scale);
    draw(img,objs,picked);
    return 0;

}