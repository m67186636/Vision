using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI.Onnx
{
    public class MediaPipeSelfieSegmentationProcessor : OnnxProcessor
    {
        protected override string RemoteModelUrl => "https://huggingface.co/qualcomm/MediaPipe-Selfie-Segmentation/resolve/main/MediaPipe-Selfie-Segmentation_float.onnx.zip?download=true";
        protected override Task PostDownloadAsync(string tempPath, string modelPath)
        {
            var directory = GetModelDirectory();
            try
            {
                var zipArchive=ZipFile.Open(tempPath, ZipArchiveMode.Read);
                foreach (var entry in zipArchive.Entries)
                {
                    if (!string.IsNullOrWhiteSpace(entry.Name))
                    {
                        var filename = Path.Combine(directory, Path.GetFileName(entry.Name));
                        entry.ExtractToFile(filename, true);
                    }
                }
                ZipFile.ExtractToDirectory(tempPath, directory, true);
            }
            catch (Exception e)
            {
                Console.WriteLine();
            }
            var files = Directory.GetFiles(directory,"*.*",SearchOption.AllDirectories);
            foreach (var f in files)
            {
                var filename = Path.Combine(directory, Path.GetFileName(f));
                File.Move(f,filename);
            }
            return Task.CompletedTask;
        }

        protected override Task ExecuteAsync(InferenceSession model, Mat image)
        {
            var inputTensor = PreExecute(image, 256);

            var inputs = new[] { NamedOnnxValue.CreateFromTensor("image", inputTensor) };
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            using var results = model.Run(inputs);
            var elapsed = stopwatch.ElapsedMilliseconds;
            var output = results.First().AsTensor<float>(); 
            var maskArray = output.ToArray();


            Mat mask = Mat.FromPixelData(256, 256, MatType.CV_32F, maskArray);
            Mat maskResized = new Mat();
            Cv2.Resize(mask, maskResized, new Size(image.Width, image.Height));
            


            // 阈值化（可选，或直接用概率做alpha混合）
            Mat alpha = new Mat();
            //Cv2.MinMaxLoc(maskResized, out double minVal, out double maxVal);
            // 归一化到[0,255]
            maskResized.ConvertTo(alpha, MatType.CV_8U, 255);


            var channels = image.Split();
            Mat[] bgra = [channels[0],channels[1],channels[2],alpha];
            Cv2.Merge(bgra, image);
            return Task.CompletedTask;
        }
    }
}
