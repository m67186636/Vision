using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI.Onnx
{
    public class U2NetPortraitProcessor : OnnxProcessor
    {
        protected override string RemoteModelUrl => "https://storage.googleapis.com/ailia-models/u2net-portrait-matting/u2net-portrait-matting.onnx";

        protected override Task ExecuteAsync(InferenceSession model, Mat image)
        {

            var inputTensor= PreExecute(image, 128);
            // Step 4: 输入OnnxRuntime
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("img", inputTensor) };


            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            using var results = model.Run(inputs);
            var elapsed = stopwatch.ElapsedMilliseconds;

            // Step 5: 获取输出mask
            var output = results.First().AsTensor<float>(); // shape: [1, 1, 512, 512]
            var maskArray = output.ToArray();
            var maskMat = Mat.FromPixelData(128, 128, MatType.CV_32F, maskArray);
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
