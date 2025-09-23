using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI.Onnx
{
    internal class PPMattingV2Processor : OnnxProcessor
    {
        protected override string RemoteModelUrl => "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/343_PP-MattingV2/resources.tar.gz";
        protected override async Task PostDownloadAsync(string tempPath, string modelPath)
        {
            await Task.Yield();
        }

        protected override Task ExecuteAsync(InferenceSession model, Mat image)
        {
            throw new NotImplementedException();
        }
    }
}
