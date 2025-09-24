using Microsoft.ML.OnnxRuntime;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI.Onnx
{
    public abstract class DnnNetProcessor:AIProcessor<Net>
    {
        internal override ColorMode ColorMode => ColorMode.RGB;
        internal override DataMode DataMode => DataMode.BCHW;
        protected override Task<Net> CreateModelAsync(string modelPath)
        {
            try
            {
                var net = CvDnn.ReadNetFromOnnx(modelPath);
                return Task.FromResult(net);
            }
            catch (Exception e)
            {
                Console.WriteLine();
            }
            return Task.FromResult<Net>(null);
        }
        protected override string GetModelPath()
        {
            return Path.Combine(Environment.CurrentDirectory, ".models", "onnxs", GetModelName(), "model.onnx");
        }
        protected override string GetModelDirectory()
        {
            return Path.Combine(Environment.CurrentDirectory, ".models", "onnxs", GetModelName());
        }
    }
}
