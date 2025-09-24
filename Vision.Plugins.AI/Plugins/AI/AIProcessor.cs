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
        internal abstract DataMode DataMode { get; }
        internal abstract ColorMode ColorMode { get; }
        protected abstract string RemoteModelUrl { get; }

        protected Lazy<Task<TModel>> LazyModelTask { get; }


        protected AIProcessor() {
            LazyModelTask= new Lazy<Task<TModel>>(InitializeModelAsync);
        }

        protected virtual async Task<TModel> InitializeModelAsync()
        {
            var modelDirectory = GetModelDirectory();
            var modelPath = GetModelPath();
            if (!Directory.Exists(modelDirectory))
                Directory.CreateDirectory(modelDirectory);
            if (!File.Exists(modelPath))
            {
                var tempPath = modelDirectory + ".download";
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
            var requestMessage=new HttpRequestMessage(HttpMethod.Get, remoteModelUrl);
            var responseMessage= await httpClient.SendAsync(requestMessage,HttpCompletionOption.ResponseHeadersRead);
            var size= responseMessage.Content.Headers.ContentLength.HasValue ? responseMessage.Content.Headers.ContentLength.Value : -1L;
            var response = await responseMessage.Content.ReadAsStreamAsync();
            using var file=File.Create(modelPath);
            await response.CopyToAsync(file);
        }

        protected abstract string GetModelPath();
        protected abstract string GetModelDirectory();
        protected string GetModelName()
        {
            var name = GetType().Name;
            if (name.EndsWith("Processor"))
                name = name.Substring(0, name.Length - "Processor".Length);
            return name;
        }
    }
}
