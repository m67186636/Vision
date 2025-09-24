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
    public class ModNetProcessor : OnnxProcessor
    {
        internal override DataMode DataMode => DataMode.BCHW;
        protected override string RemoteModelUrl => "https://storage.googleapis.com/ailia-models/modnet/modnet.opt.onnx";

        //protected override Task ExecuteAsync(Net model, Mat image)
        //{
        //    // MODNet要求输入为 512x512, RGB, float32
        //    using var resized = new Mat();
        //    Cv2.Resize(image, resized, new Size(512, 512));
        //    Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

        //    using var blob = CvDnn.BlobFromImage(resized, 1.0 / 255.0, new Size(512, 512));
        //    model.SetInput(blob);

        //    using var output = model.Forward(); // [1,1,512,512] float32

        //    Mat mask = Mat.FromPixelData(512, 512, MatType.CV_32F, output.Ptr(0));
        //    Cv2.Normalize(mask, mask, 0, 255, NormTypes.MinMax);
        //    mask.ConvertTo(mask, MatType.CV_8U);

        //    using var maskResized = new Mat();
        //    Cv2.Resize(mask, maskResized, image.Size());

        //    Mat[] channels = Cv2.Split(image);
        //    Mat rgba = new Mat();
        //    Cv2.Merge(new Mat[] { channels[0], channels[1], channels[2], maskResized }, image);

        //    foreach (var c in channels) c.Dispose();
        //    return Task.CompletedTask;
        //}
        protected override Task ExecuteAsync(InferenceSession model, Mat image)
        {
            //var metadata=model.InputMetadata.FirstOrDefault().Value;

            var r = (float)Math.Max(image.Height, image.Width) / 512;
            var w = (int)(image.Width / r);
            var h = (int)(image.Height / r);
            var inputTensor = PreExecute(image, w,h);
            // Step 4: 输入OnnxRuntime
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            using var results = model.Run(inputs);
            var elapsed = stopwatch.ElapsedMilliseconds;
            // Step 5: 获取输出mask
            var output = results.First().AsTensor<float>(); // shape: [1, 1, 512, 512]
            var maskArray = output.ToArray();
            var maskMat = Mat.FromPixelData(h, w, MatType.CV_32F, maskArray);
            //var maskMat = new Mat(512, 512, MatType.CV_32F, maskArray);


            var maskResized = new Mat();
            Cv2.Resize(maskMat, maskResized, image.Size());
            maskResized = maskResized.Clone();
            maskResized.ConvertTo(maskResized, MatType.CV_8U, 255);

            // Step 7: 合成透明PNG
            Mat[] channelsArr = Cv2.Split(image);
            Cv2.Merge(new Mat[] { channelsArr[0], channelsArr[1], channelsArr[2], maskResized }, image);
            foreach (var c in channelsArr) c.Dispose();
            return Task.CompletedTask;
        }
        protected  Task ExecuteAsync2(InferenceSession model, Mat image)
        {

            // Step 1: Resize为512x512，BGR转RGB
            using var resized = new Mat();
            Cv2.Resize(image, resized, new Size(512, 512));
            Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

            // Step 2: 转为float32，归一化到[0,1]
            var imgFloat = new Mat();
            resized.ConvertTo(imgFloat, MatType.CV_32F, 1.0 / 255);

            // Step 3: NHWC转NCHW
            int width = imgFloat.Width;
            int height = imgFloat.Height;
            int channels = imgFloat.Channels(); // 3

            var inputTensor = new DenseTensor<float>(new[] { 1, channels, height, width });
            float[] inputData = new float[channels * height * width];

            int idx = 0;
            for (int c = 0; c < channels; c++)
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        inputData[idx++] = imgFloat.At<Vec3f>(y, x)[c];

            inputData.CopyTo(inputTensor.Buffer.Span);

            // Step 4: 输入OnnxRuntime
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
            using var results = model.Run(inputs);

            // Step 5: 获取输出mask
            var output = results.First().AsTensor<float>(); // shape: [1, 1, 512, 512]
            var maskArray = output.ToArray();
            var maskMat = Mat.FromPixelData(512, 512, MatType.CV_32F, maskArray);
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
