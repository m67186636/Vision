using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace Vision.Plugins.AI.Onnx
{
    public class RobustVideoMattingProcessor : OnnxProcessor
    {
        const string ResNet50Url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx";
        const string MobileNetV2Url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx";
        protected override string RemoteModelUrl => MobileNetV2Url;

        private Tensor<float> R1i { get; set; }
        private Tensor<float> R2i { get; set; }
        private Tensor<float> R3i { get; set; }
        private Tensor<float> R4i { get; set; }

        public RobustVideoMattingProcessor() {

            R1i = InitMemory(1, 1, 1);
            R2i = InitMemory(1, 1, 1);
            R3i = InitMemory(1, 1, 1);
            R4i = InitMemory(1, 1, 1);
        }
        Tensor<float> InitMemory(int c, int h, int w)
            => new DenseTensor<float>(new float[c * h * w], new[] { 1, c, h, w });
        protected override Task ExecuteAsync(InferenceSession model, Mat image)
        {
            var r = (float)Math.Max(image.Height, image.Width) / 512;
            var width = (int)(image.Width / r);
            var height = (int)(image.Height / r);
            var srcTensor = PreExecute(image, width, height);
            var ratioTensor = new DenseTensor<float>(new float[] { 0.25f }, new[] { 1 });
            var binding=model.CreateIoBinding();
            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("src", srcTensor),
            NamedOnnxValue.CreateFromTensor("r1i", R1i),
            NamedOnnxValue.CreateFromTensor("r2i", R2i),
            NamedOnnxValue.CreateFromTensor("r3i", R3i),
            NamedOnnxValue.CreateFromTensor("r4i", R4i),
            NamedOnnxValue.CreateFromTensor("downsample_ratio", ratioTensor)
        };
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            using var results = model.Run(inputs);
            var elapsed = stopwatch.ElapsedMilliseconds;
            var fgr = results.First(x => x.Name == "fgr").AsTensor<float>(); // [1,3,H,W]
            var pha = results.First(x => x.Name == "pha").AsTensor<float>(); // [1,1,H,W]
            var r1o = results.First(x => x.Name == "r1o").AsTensor<float>();
            var r2o = results.First(x => x.Name == "r2o").AsTensor<float>();
            var r3o = results.First(x => x.Name == "r3o").AsTensor<float>();
            var r4o = results.First(x => x.Name == "r4o").AsTensor<float>();
            R1i = r1o.Clone(); 
            R2i = r2o.Clone();
            R3i = r3o.Clone();
            R4i = r4o.Clone();
            var maskArray = pha.ToArray();
            var maskMat = Mat.FromPixelData(height,width, MatType.CV_32F, maskArray);

            maskMat.ConvertTo(maskMat, MatType.CV_8U, 255);
            Cv2.Normalize(maskMat, maskMat, 0, 255, NormTypes.MinMax);
            maskMat.ConvertTo(maskMat, MatType.CV_8U);

            using var maskResized = new Mat();
            Cv2.Resize(maskMat, maskResized, image.Size());


            int featherRadius = Math.Clamp(Math.Min(image.Width, image.Height) / 300, 3, 18);

            using var feathered = FeatherAlpha(maskResized, featherRadius);

            // Step 7: 合成透明PNG
            Mat[] channelsArr = Cv2.Split(image);
            Cv2.Merge(new Mat[] { channelsArr[0], channelsArr[1], channelsArr[2], feathered }, image);
            foreach (var c in channelsArr) c.Dispose();

            return Task.CompletedTask;
        }

        // 基础羽化：高斯 + 核心保护，适合快速平滑边缘
        private Mat FeatherAlpha(Mat alpha8u, int radius)
        {
            radius = Math.Max(1, radius);
            int k = radius * 2 + 1;

            var alphaFloat = new Mat();
            alpha8u.ConvertTo(alphaFloat, MatType.CV_32F, 1.0 / 255.0);

            // 先模糊
            Cv2.GaussianBlur(alphaFloat, alphaFloat, new Size(k, k), radius * 0.5, radius * 0.5, BorderTypes.Default);

            // 保留接近 1 的核心，防止主体整体被软化
            using var core = new Mat();
            Cv2.Threshold(alphaFloat, core, 0.995, 1.0, ThresholdTypes.Binary);
            Cv2.Max(alphaFloat, core, alphaFloat);

            // 转回 8U
            var outAlpha = new Mat();
            alphaFloat.ConvertTo(outAlpha, MatType.CV_8U, 255.0);

            alphaFloat.Dispose();
            return outAlpha;
        }

        private void Save(Tensor<float> fgr, string filename, int width, int height)
        {
            var fgrArray = fgr.ToArray(); // 长度为 1*3*H*W

            // 创建一个新的 float[H, W, 3] 数组
            float[] hwc = new float[height * width * 3];

            int idx = 0;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        // 原数组索引：[c * H * W + h * W + w]
                        hwc[idx++] = fgrArray[c * height * width + h * width + w];
                    }
                }
            }
            Mat mat = Mat.FromPixelData(height, width,MatType.CV_32FC3, hwc);

            mat.ConvertTo(mat, MatType.CV_8UC3, 255);

            // 如需BGR保存，需转换颜色
            Cv2.CvtColor(mat, mat, ColorConversionCodes.RGB2BGR);
            // 保存为 PNG 或 JPG
            mat.SaveImage("fgr.png");
        }
    }
}
