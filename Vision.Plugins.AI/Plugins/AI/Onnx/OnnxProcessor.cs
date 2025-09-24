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
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace Vision.Plugins.AI.Onnx
{
    public abstract class OnnxProcessor : AIProcessor<InferenceSession>
    {
        internal override ColorMode ColorMode => ColorMode.RGB;
        internal override DataMode DataMode => DataMode.BCHW;
        protected OnnxProcessor() { 
        }

        protected override Task<InferenceSession> CreateModelAsync(string modelPath)
        {
            var inferenceSession=new InferenceSession(modelPath,AIOptions.SessionOptions);
            return Task.FromResult(inferenceSession);
        }


        protected DenseTensor<float> PreExecute(Mat image, int width, int height)
        {

            // Step 1: Resize为width*height，BGR转RGB
            using var resized = new Mat();
            Cv2.Resize(image, resized, new Size(width, height));
            if(ColorMode== ColorMode.RGB)
                Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

            var sharp = GetShape(resized);
            Mat blob = BlobFromImage(resized, sharp);

            var inputTensorData = new float[blob.Total()];
            Marshal.Copy(blob.Data, inputTensorData, 0, inputTensorData.Length);
            
            var inputTensor = new DenseTensor<float>(inputTensorData, sharp);
            return inputTensor;
        }

        private int[] GetShape(Mat mat)
        {
            switch (DataMode)
            {
                case DataMode.BCHW:
                    return [1, mat.Channels(), mat.Height, mat.Width];
                case DataMode.BCWH:
                    return [1, mat.Channels(), mat.Width, mat.Height];
                default:
                    throw new NotImplementedException();
            }
        }

        private Mat BlobFromImage(Mat mat, int[] sharp)
        {
            if (!mat.IsContinuous())
                mat = mat.Clone();
            var data= CvDnn.BlobFromImage(mat,1.0/255);
            switch (DataMode)
            {
                case DataMode.BCHW:
                    return data; 
                case DataMode.BCWH:
                    return DataUtils.BCHW_to_BCWH(data);
                default:
                    throw new NotImplementedException();
            }
        }

        protected DenseTensor<float> PreExecute(Mat image, int size)
        {
            return PreExecute(image, size, size);
        }
        protected override string GetModelPath()
        {
            return Path.Combine(GetModelDirectory(), "model.onnx");
        }
        protected override string GetModelDirectory()
        {
            return Path.Combine(Environment.CurrentDirectory, ".models", "onnxs", GetModelName());
        }
    }
}
