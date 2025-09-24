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
using Vision.Plugins.Basic;

namespace Vision.Plugins.AI.Onnx
{
    public class BackgroundMattingV2Processor : OnnxProcessor
    {
        const string ResNet50Url = "https://storage.googleapis.com/ailia-models/background_matting_v2/resnet50.onnx";
        const string MobileNetV2Url = "https://storage.googleapis.com/ailia-models/background_matting_v2/mobilenetv2.onnx";
        protected override string RemoteModelUrl => MobileNetV2Url;

        protected DenseTensor<float> BgrTensor { get; }

        public BackgroundMattingV2Processor() {
            BgrTensor = CreateBgrTensor(1,1,35, 198, 50);
        }

        private DenseTensor<float> CreateBgrTensor(int width, int height, byte r, byte g, byte b)
        {
            return CreateBgrTensor(width,height,new Scalar(b, g, r));
        }
        private DenseTensor<float> CreateBgrTensor(int width, int height, Scalar color)
        {
            using var mat = new Mat(height, width, MatType.CV_8UC3, color);
            Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
            Mat blob = CvDnn.BlobFromImage(mat, 1.0/255, new Size(width, height)); // shape: [1,3,width,height], type: CV_32F
            float[] inputTensorData = new float[3 * width * height];
            Marshal.Copy(blob.Ptr(0), inputTensorData, 0, inputTensorData.Length);
            var inputTensor = new DenseTensor<float>(inputTensorData, new[] { 1, 3, width, height });
            return inputTensor;
        }

        protected override Task ExecuteAsync(InferenceSession model, Mat image)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var r = (float)Math.Max(image.Height, image.Width) / 960;
            var w= (int)(image.Width / r);  
            var h= (int)(image.Height / r);
            stopwatch.Restart();
            var srcTensor = PreExecute(image, w,h);
            var elapsed = stopwatch.ElapsedMilliseconds;
            var pixel = ChromaKeyUtils.DetectMainColor(image);
            //var bgrTensor = CreateBgrTensor(w, h, pixel[2], pixel[1], pixel[0]);
            var bgrTensor = CreateBgrTensor(w, h, pixel);
            // Step 4: 输入OnnxRuntime
            var inputs = new[] {
                NamedOnnxValue.CreateFromTensor("src", srcTensor),
                NamedOnnxValue.CreateFromTensor("bgr", srcTensor)
            };
            stopwatch.Restart();
            using var results = model.Run(inputs);
            elapsed = stopwatch.ElapsedMilliseconds;
            // Step 5: 获取输出mask
            var output = results.First().AsTensor<float>(); // shape: [1, 1, 512, 512]
            var maskArray = output.ToArray();
            var maskMat = Mat.FromPixelData(h, w, MatType.CV_32F, maskArray);
            //var maskMat = new Mat(512, 512, MatType.CV_32F, maskArray);

            // Step 6: 归一化、转uint8
            Cv2.Normalize(maskMat, maskMat, 0, 255, NormTypes.MinMax);
            maskMat.ConvertTo(maskMat, MatType.CV_8U);

            using var maskResized = new Mat();
            Cv2.Resize(maskMat, maskResized, image.Size());

            // Step 7: 合成透明PNG
            Mat[] channelsArr = Cv2.Split(image);
            Cv2.Merge(new Mat[] { channelsArr[0], channelsArr[1], channelsArr[2], maskResized }, image);
            foreach (var c in channelsArr) c.Dispose();
            return Task.CompletedTask;
        }
    }
}
