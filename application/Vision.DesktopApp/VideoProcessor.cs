using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Vision.Plugins.Basic;
using Size = OpenCvSharp.Size;

namespace Vision
{
    public class VideoProcessor
    {
        private readonly Mat backgroundFrame = new();
        private readonly Mat foregroundFrame = new();
        private Mat _nextBackgroundFrame;
        private Mat _nextForegroundFrame;
        private bool mainValid;
        private double blurAlpha=0d;
        private double blurStep=0.05d;

        private IProcessor Processor { get; }
        public event Action<Bitmap, Bitmap, Bitmap> OnFrameCaptured;

        public VideoProcessor()
        {
            Processor = new ChromaKeyProcessor();
        }
        public void SetNextBackgroundFrame(Mat frame)
        {
            lock (this)
                _nextBackgroundFrame = frame;
        }
        public void SetNextForegroundFrame(Mat frame)
        {
            lock(this)
                _nextForegroundFrame = frame;
        }
        public async Task ExecuteAsync()
        {
            while (true)
            {
                if (_nextBackgroundFrame == null && _nextForegroundFrame == null) { 
                    await Task.Delay(10);
                    continue;
                }
                lock (this)
                {
                    _nextBackgroundFrame?.CopyTo(backgroundFrame);
                    _nextForegroundFrame?.CopyTo(foregroundFrame);
                }
                if (backgroundFrame.Empty() || foregroundFrame.Empty())
                {
                    await Task.Delay(10);
                    continue;
                }

                await Processor.ExecuteAsync(_nextForegroundFrame);
                mainValid = _nextForegroundFrame != null && !_nextForegroundFrame.Empty();
                if (mainValid)
                {
                    var rate = GetAlphaRate(foregroundFrame);
                    mainValid = rate > 0.15;
                }
                if (mainValid)
                    blurAlpha = Math.Min(1.0, blurAlpha + blurStep);
                else
                    blurAlpha = Math.Max(0.0, blurAlpha - blurStep);

                var mat = await ExecuteAsync(backgroundFrame, foregroundFrame, blurAlpha);
                OnFrameCaptured?.Invoke(BitmapConverter.ToBitmap(mat), BitmapConverter.ToBitmap(backgroundFrame), BitmapConverter.ToBitmap(foregroundFrame));
            }
        }


        private Task<Mat> ExecuteAsync( Mat backgroundFrame, Mat foregroundFrame, double blurAlpha)
        {
            if (backgroundFrame.Empty()) return Task.FromResult(foregroundFrame);


            var blurred = new Mat();
            if (blurAlpha > 0)
                Cv2.GaussianBlur(backgroundFrame, blurred, new Size(35, 35), 0);
            else
                blurred = backgroundFrame;


            Mat mixed = new Mat();
            if (blurAlpha > 0)
                Cv2.AddWeighted(backgroundFrame, 1 - blurAlpha, blurred, blurAlpha, 0, mixed);
            else
                mixed = backgroundFrame;
            if (foregroundFrame.Type() == MatType.CV_8UC4)
                return Task.FromResult(OverlayBGRAOnBGR(mixed.Clone(), foregroundFrame));
            return Task.FromResult(mixed);
        }

        private Mat OverlayBGRAOnBGR(Mat background, Mat foreground)
        {

            var stopwatch = Stopwatch.StartNew();
            if (background.Size() != foreground.Size())
                Cv2.Resize(foreground, foreground, background.Size());
            var result = new Mat();
            Cv2.CvtColor(background, result, ColorConversionCodes.BGR2BGRA);

            // 分离通道
            var foregroundChannels = Cv2.Split(foreground);
            var foregroundAalpha = new Mat();
            foregroundChannels[3].ConvertTo(foregroundAalpha, MatType.CV_32F, 1.0 / 255);
            using var foregroundBChannel = ConvertChannelFrom(foregroundChannels[0], MatType.CV_32F);
            using var foregroundGChannel = ConvertChannelFrom(foregroundChannels[1], MatType.CV_32F);
            using var foregroundRChannel = ConvertChannelFrom(foregroundChannels[2], MatType.CV_32F);


            var backgroundChannels = Cv2.Split(result);
            using var backgroundBChannel = ConvertChannelFrom(backgroundChannels[0], MatType.CV_32F);
            using var backgroundGChannel = ConvertChannelFrom(backgroundChannels[1], MatType.CV_32F);
            using var backgroundRChannel = ConvertChannelFrom(backgroundChannels[2], MatType.CV_32F);


            // fgAlpha 是归一化的 float32 Mat，值在 [0, 1]
            using var oneMat = Mat.Ones(foregroundAalpha.Size(), foregroundAalpha.Type()); // 全1矩阵
            using var invAlpha = new Mat();
            Cv2.Subtract(oneMat, foregroundAalpha, invAlpha); // invAlpha = 1 - fgAlpha


            using Mat outBChannel = foregroundBChannel.Mul(foregroundAalpha) + backgroundBChannel.Mul(invAlpha);
            using Mat outGChannel = foregroundGChannel.Mul(foregroundAalpha) + backgroundGChannel.Mul(invAlpha);
            using Mat outRChannel = foregroundRChannel.Mul(foregroundAalpha) + backgroundRChannel.Mul(invAlpha);

            outBChannel.ConvertTo(outBChannel, MatType.CV_8U);
            outGChannel.ConvertTo(outGChannel, MatType.CV_8U);
            outRChannel.ConvertTo(outRChannel, MatType.CV_8U);
            Cv2.Merge([outBChannel, outGChannel, outRChannel], result);
            foreach (var mat in foregroundChannels) mat.Dispose();
            foreach (var mat in backgroundChannels) mat.Dispose();
            var elapsed = stopwatch.ElapsedMilliseconds;
            return result;
        }
        private Mat ConvertChannelFrom(Mat channel, MatType type)
        {
            Mat mat = new Mat();
            channel.ConvertTo(mat, type);
            return mat;
        }

        private double GetAlphaRate(Mat frame)
        {
            if (frame.Type() != MatType.CV_8UC4)
                return 0;

            var total = frame.Width * frame.Height;

            Mat alpha = new Mat();
            Cv2.ExtractChannel(frame, alpha, 3);

            using var mask = new Mat();
            Cv2.Threshold(alpha, mask, 50, 255, ThresholdTypes.Binary);
            var alphaTotal = Cv2.CountNonZero(mask);
            return (double)alphaTotal / total;
        }
    }
}
