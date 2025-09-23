using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI.Onnx
{
    public abstract class OnnxProcessor : AIProcessor<InferenceSession>
    {
        protected OnnxProcessor() { 
        }

        protected override Task<InferenceSession> CreateModelAsync(string modelPath)
        {
            var inferenceSession=new InferenceSession(modelPath,AIOptions.SessionOptions);
            return Task.FromResult(inferenceSession);
        }


        protected DenseTensor<float> PreExecute(Mat image, int width, int height)
        {

            // Step 1: Resize为width*height，BGR转RGB
            using var resized = new Mat();
            Cv2.Resize(image, resized, new Size(width, height));
            resized.SaveImage("src.jpg");
            Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);
            // Step 2: 转为float32，归一化到[0,1]
            //var imgFloat = new Mat();
            resized.ConvertTo(resized, MatType.CV_32F, 1.0 / 255);

            // Step 3: NHWC转NCHW
            Mat blob = CvDnn.BlobFromImage(resized, 1.0, new Size(width, height), new Scalar(0, 0, 0), false, false); // shape: [1,3,width,height], type: CV_32F
            float[] inputTensorData = new float[3 * height * width];
            Marshal.Copy(blob.Data, inputTensorData, 0, inputTensorData.Length);
            var inputTensor = new DenseTensor<float>(inputTensorData, new[] { 1, 3,  height, width });
            return inputTensor;
        }
        protected DenseTensor<float> PreExecute(Mat image, int size)
        {
            return PreExecute(image, size, size);
        }
        protected override string GetModelPath()
        {
            return Path.Combine(Environment.CurrentDirectory, ".models", "onnxs", GetModelName() + ".onnx");
        }
    }
}
