using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI.Onnx
{
    public class U2NetProcessor: DnnNetProcessor
    {
        protected override string RemoteModelUrl => "https://storage.googleapis.com/ailia-models/u2net/u2net.onnx";

        protected override Task ExecuteAsync(Net model, Mat image)
        {
            // Step 1: Resize to 320x320, 转RGB
            Mat resized = new Mat();
            Cv2.Resize(image, resized, new Size(320, 320));
            Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

            // Step 2: 构造DNN输入
            Mat blob = CvDnn.BlobFromImage(resized, 1.0 / 255.0, new Size(320, 320), new Scalar(), swapRB: false, crop: false);

            // Step 3: 前向推理
            model.SetInput(blob);
            Mat output = model.Forward();

            // Step 4: 输出mask，归一化 [1,1,320,320] -> [320,320]
            
            Mat mask = Mat.FromPixelData(320, 320, MatType.CV_32F, output.Ptr(0));
            Cv2.Normalize(mask, mask, 0, 255, NormTypes.MinMax);
            mask.ConvertTo(mask, MatType.CV_8U);

            // Step 5: mask缩放回原尺寸
            Mat maskResized = new Mat();
            Cv2.Resize(mask, maskResized, image.Size());

            // Step 6: 合成透明PNG
            Mat[] channels = Cv2.Split(image);
            Mat rgba = new Mat();
            Cv2.Merge([channels[0], channels[1], channels[2], maskResized], image);
            //rgba.SetTo(image);
            //return rgba;
            return Task.CompletedTask;
        }

    }
}
