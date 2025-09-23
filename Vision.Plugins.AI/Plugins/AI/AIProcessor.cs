using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI
{
    public interface IAIProcessor : IProcessor
    {
    }
    public abstract class AIProcessor<TModel>: IAIProcessor
    {
        protected abstract string RemoteModelUrl { get; }

        protected Lazy<Task<TModel>> LazyModelTask { get; }


        protected AIProcessor() {
            LazyModelTask= new Lazy<Task<TModel>>(InitializeModelAsync);
        }

        protected virtual async Task<TModel> InitializeModelAsync()
        {
            var modelPath = GetModelPath();
            var directory = Path.GetDirectoryName(modelPath);
            if (!Directory.Exists(directory))
                Directory.CreateDirectory(directory);
            if (!File.Exists(modelPath))
            {
                var tempPath = modelPath + ".download";
                await DownloadAsync(RemoteModelUrl, tempPath);
                await PostDownloadAsync(tempPath,modelPath);
            }
            return await CreateModelAsync(modelPath);
        }

        protected virtual async Task PostDownloadAsync(string tempPath, string modelPath)
        {
            await Task.Run(()=> File.Move(tempPath, modelPath));
        }

        protected abstract Task<TModel> CreateModelAsync(string modelPath);

        public async Task ExecuteAsync(Mat image) {
            if (!LazyModelTask.IsValueCreated)
            {
                _= LazyModelTask.Value;
                Cv2.PutText(image, "Model Loading...", new Point(0, 50), HersheyFonts.Italic, 2, new Scalar(255, 0, 0));
            }
            else if (!LazyModelTask.Value.IsCompleted)
            {
                Cv2.PutText(image, "Model Loading...", new Point(0, 50), HersheyFonts.Italic, 2, new Scalar(255, 0, 0));
            }
            else if (LazyModelTask.Value.IsCompletedSuccessfully)
            {
                var model = await LazyModelTask.Value;
                try
                {
                    await ExecuteAsync(model, image);
                }
                catch (Exception e)
                {
                    Console.WriteLine();
                }
            }
            else
            {
                Cv2.PutText(image, "Model Loading failed", new Point(0, 50), HersheyFonts.Italic, 2, new Scalar(255, 0, 0));
            }
        }

        protected abstract Task ExecuteAsync(TModel model, Mat image);

        protected async Task DownloadAsync(string remoteModelUrl, string modelPath)
        {
            using var httpClient = new HttpClient();
            var response = await httpClient.GetStreamAsync(remoteModelUrl);
            using var file=File.Create(modelPath);
            await response.CopyToAsync(file);
        }

        protected abstract string GetModelPath();
        protected string GetModelName()
        {
            var name = GetType().Name;
            if (name.EndsWith("Processor"))
                name = name.Substring(0, name.Length - "Processor".Length);
            return name;
        }
    }
}
