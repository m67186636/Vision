using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Vision.Plugins.AI;

namespace Vision
{
    public class VideoCaptureHelper
    {
        public static VideoCaptureHelper Instence { get; }

        static VideoCaptureHelper() { Instence = new(); }
        protected VideoCapture MainCapture { get; }
        protected IProcessor Processor { set; get; }

        public event Action<Bitmap,Bitmap> OnFrameCaptured;
        private VideoCaptureHelper()
        {
            var modules=Assembly.GetEntryAssembly().GetReferencedAssemblies();
            AIOptions.SessionOptions.IntraOpNumThreads = Environment.ProcessorCount;
             MainCapture = new VideoCapture(0); // 默认摄像头
        }

        public async Task StartAsync()
        {
            await Task.Run(async () =>
            {
                while (MainCapture.IsOpened())
                {
                    using var before = new Mat();
                    MainCapture.Read(before);
                    if (before.Empty()) continue;
                    Cv2.Flip(before, before, FlipMode.Y); // 或 FlipMode.Horizontal
                    using var after = before.Clone();
                    await ExecuteAsync(after);
                    OnFrameCaptured?.Invoke(BitmapConverter.ToBitmap(before), BitmapConverter.ToBitmap(after));
                }
            });
        }
        public void SetProcessor<TProcessor>()
            where TProcessor : IProcessor ,new()
        {
            var options = new OrtCUDAProviderOptions();
            var s=options.GetOptions();
            AIOptions.SessionOptions.AppendExecutionProvider_CUDA(options);
            Processor = new TProcessor();
        }

        private async Task ExecuteAsync(Mat mat)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            if (Processor != null)
                await Processor.ExecuteAsync(mat);
            var elapsedMilliseconds = stopwatch.ElapsedMilliseconds;
            Cv2.PutText(mat, $"{elapsedMilliseconds}(ms)", new OpenCvSharp.Point(0,mat.Height/10), HersheyFonts.Italic, mat.Height/100, new Scalar(255, 255, 255));
        }
    }
}
