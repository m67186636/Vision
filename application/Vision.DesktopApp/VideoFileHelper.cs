using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Vision.Plugins.AI;
using Vision.Plugins.AI.Onnx;
using Vision.Plugins.Basic;
using static System.Resources.ResXFileRef;
using Size = OpenCvSharp.Size;

namespace Vision
{
    public class VideoFileHelper
    {
        public static VideoFileHelper Instance { get; }

        static VideoFileHelper() { Instance = new(); }
        private IProcessor Processor { get; }
        public event Action<Bitmap,Bitmap> OnFrameCaptured;
        public VideoFileHelper()
        {
            Processor = new ChromaKeyProcessor();
        }

        public async Task StartAsync()
        {
            await Task.Run(async () =>
            {
                var backgroundVideoPath = @"B:\lives\kiwifruit\backgroud\VID_20250820_114440_720P1.mp4";
                var mainVideoDirectory = @"B:\lives\kiwifruit\videos";
                var mainVideos = Directory.GetFiles(mainVideoDirectory,"*.mp4");
                var second = 2000.0;
                using var backgroundVideo = new VideoCapture(backgroundVideoPath);
                if (!backgroundVideo.IsOpened())
                    return;

                var backgroundVideoFPS = backgroundVideo.Fps > 0 ? backgroundVideo.Fps : 25;
                var backgroundVideoInterval = second / backgroundVideoFPS;


                var mainVideoIndex = 0;
                var mainVideo = new VideoCapture(mainVideos[mainVideoIndex]);

                if (!mainVideo.IsOpened())
                    return;
                mainVideo.Set(VideoCaptureProperties.PosFrames, 20);
                var mainVideoFPS = mainVideo.Fps > 0 ? mainVideo.Fps : 25;
                var mainVideoInterval = second / mainVideoFPS;

               
                var backgroundFrame = new Mat();
                var mainFrame = new Mat();
                var backgroundNextTime = (double)Stopwatch.GetTimestamp() * 1000 / Stopwatch.Frequency;
                var mainNextTime = (double)Stopwatch.GetTimestamp() * 1000 / Stopwatch.Frequency;
                var mainValid = false; 
                var blurAlpha = 0d;                  // 当前模糊强度
                var blurStep = 0.05d;
                var stopwatch = Stopwatch.StartNew();
                while (true)
                {
                    try
                    {
                        var needProcess = false;
                        var now = (double)Stopwatch.GetTimestamp() * 1000 / Stopwatch.Frequency;
                        if (now >= backgroundNextTime)
                        {
                            if (!backgroundVideo.Read(backgroundFrame) || backgroundFrame.Empty())
                            {
                                backgroundVideo.Set(VideoCaptureProperties.PosFrames, 0);
                                backgroundVideo.Read(backgroundFrame);
                            }
                            backgroundNextTime += backgroundVideoInterval;
                            needProcess = true;
                        }

                        if (now >= mainNextTime)
                        {
                            if (!mainVideo.Read(mainFrame) || mainFrame.Empty())
                            {
                                mainVideo.Release();
                                mainVideoIndex = (mainVideoIndex + 1) % mainVideos.Length;
                                mainVideo = new VideoCapture(mainVideos[mainVideoIndex]);
                                mainVideo.Set(VideoCaptureProperties.PosFrames, 20);
                                mainVideoFPS = mainVideo.Fps > 0 ? mainVideo.Fps : 25;
                                mainVideoInterval = second / mainVideoFPS;
                                mainVideo.Read(mainFrame);
                            }
                            mainNextTime += mainVideoInterval;
                            needProcess = true;

                            await Processor.ExecuteAsync(mainFrame);
                            var elapsed = stopwatch.ElapsedMilliseconds;
                            mainValid = mainFrame != null && !mainFrame.Empty();
                            if (mainValid)
                            {
                                var rate = GetAlphaRate(mainFrame);
                                mainValid = rate > 0.15;
                            }
                        }
                        if (mainValid)
                            blurAlpha = Math.Min(1.0, blurAlpha + blurStep);
                        else
                            blurAlpha = Math.Max(0.0, blurAlpha - blurStep);
                        if (needProcess)
                        {
                            var mat = await ExecuteAsync(mainFrame.Clone(), backgroundFrame.Clone(), blurAlpha);
                            OnFrameCaptured?.Invoke(BitmapConverter.ToBitmap(mat),BitmapConverter.ToBitmap(mainFrame));
                            var elapsed = stopwatch.ElapsedMilliseconds;
                            Console.WriteLine();
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine();
                    }
                    stopwatch.Restart();
                }
            });
        }

        private double GetAlphaRate(Mat frame)
        {
            if (frame.Type() != MatType.CV_8UC4)
                return 0;

            var total = frame.Width* frame.Height;

            Mat alpha = new Mat();
            Cv2.ExtractChannel(frame, alpha, 3);

            using var mask = new Mat();
            Cv2.Threshold(alpha, mask, 50, 255, ThresholdTypes.Binary);
            var alphaTotal = Cv2.CountNonZero(mask);
            return (double)alphaTotal / total;
        }

        private Task<Mat> ExecuteAsync(Mat mainFrame, Mat backgroundFrame, double blurAlpha)
        {
            if (backgroundFrame.Empty()) return Task.FromResult(mainFrame);


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
            //Mat output = mixed.Clone();
            //Cv2.AddWeighted(mainFrame, 1, output, 1, 0, output);
            if (mainFrame.Type() == MatType.CV_8UC4)
            return Task.FromResult(OverlayBGRAOnBGR(mixed.Clone(),mainFrame));
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

            outBChannel.ConvertTo(outBChannel,MatType.CV_8U);
            outGChannel.ConvertTo(outGChannel, MatType.CV_8U);
            outRChannel.ConvertTo(outRChannel, MatType.CV_8U);
            Cv2.Merge([outBChannel,outGChannel,outRChannel], result);
            foreach (var mat in foregroundChannels) mat.Dispose();
            foreach (var mat in backgroundChannels) mat.Dispose();
            var elapsed = stopwatch.ElapsedMilliseconds;
            return result;


            //using (Mat mask = new Mat())
            //{
            //    Cv2.ExtractChannel(bgra, mask, 3); // alpha通道

            //    //for (int y = 0; y < mask.Height; y++)
            //    //{
            //    //    for (int x = 0; x < mask.Width; x++)
            //    //    {
            //    //        var v = mask.At<byte>(y, x);
            //    //        if (v > 50)
            //    //            mask.Set(y, x, (byte)255);
            //    //    }
            //    //}
            //    //Cv2.Threshold(mask, mask, 100, 255, ThresholdTypes.Binary); // 只在alpha > 100 的地方mask为255
            //                                                                // 只在alpha>0的地方将前景拷贝到背景
            //    var stopwatch = Stopwatch.StartNew();
            //    for (int y = 0; y < mask.Height; y++) { 
            //        for (int x = 0; x < mask.Width; x++)
            //        {
            //            var a = mask.At<byte>(y, x);
            //            var alpha = a / 255.0f;
            //            if (alpha == 0) { }
            //            else if (alpha < 1)
            //            {

            //                var bgPixel = bgr.At<Vec4b>(y, x);
            //                var fgPixel = bgra.At<Vec4b>(y, x);
            //                byte b = (byte)(fgPixel[0] * alpha + bgPixel[0] * (1 - alpha));
            //                byte g = (byte)(fgPixel[1] * alpha + bgPixel[1] * (1 - alpha));
            //                byte r = (byte)(fgPixel[2] * alpha + bgPixel[2] * (1 - alpha));
            //                mask.Set(y, x, (byte)255);
            //                result.Set(y, x, new Vec4b(b, g, r, 255));
            //            }
            //            else if (alpha ==1)
            //            {
            //                var fgPixel = bgra.At<Vec4b>(y, x);
            //                result.Set(y, x, fgPixel);
            //            }
            //        }
            //    }
            //    var elapsed = stopwatch.ElapsedMilliseconds;
            //    Console.WriteLine();
            //    //bgra.CopyTo(result, mask);
            //    //Cv2.AddWeighted(result, 1, bgra, 1, 0,result);
            //}
            //return result;
        }

        private Mat ConvertChannelFrom(Mat channel, int cV_32F)
        {
            Mat mat = new Mat();
            channel.ConvertTo(mat, MatType.CV_32F);
            return mat;
        }

        private Mat OverlayBGRAOnBGR1(Mat bgr, Mat bgra)
        {
            
            // 检查尺寸
            if (bgr.Size() != bgra.Size())
                bgra.Resize(bgr.Size());
            if (bgr.Type() != MatType.CV_8UC3)
                throw new ArgumentException("bgr 必须是 CV_8UC3");
            if (bgra.Type() != MatType.CV_8UC4)
                throw new ArgumentException("bgra 必须是 CV_8UC4");
            //Cv2.CvtColor(bgr, bgr, ColorConversionCodes.BGR2BGRA);
            // 分离BGRA的通道
            Mat[] bgraChannels = Cv2.Split(bgra);
            Mat alpha = bgraChannels[3]; // alpha通道
            for (int y = 0; y < alpha.Height; y++)
            {
                for (int x = 0; x < alpha.Width; x++)
                {
                    var v = alpha.At<byte>(y, x);
                    if (v > 50)
                        alpha.Set(y, x, (byte)255);
                    alpha.Set(y, x, (byte)0);
                }
            }

            // 将alpha归一化到0~1，并扩展到三通道
            Mat alphaF = new Mat();
            alpha.ConvertTo(alphaF, MatType.CV_32FC1, 1.0 / 255);
            Mat[] alphaFs = { alphaF, alphaF, alphaF };
            Mat alpha3 = new Mat();
            Cv2.Merge(alphaFs, alpha3);

            // 前景BGR和背景BGR转为float
            Mat fgBGR = new Mat();
            Cv2.Merge(new Mat[] { bgraChannels[0], bgraChannels[1], bgraChannels[2] }, fgBGR);
            Mat fgBGRf = new Mat();
            fgBGR.ConvertTo(fgBGRf, MatType.CV_32FC3, 1.0 / 255);
            Mat bgrF = new Mat();
            bgr.ConvertTo(bgrF, MatType.CV_32FC3, 1.0 / 255);

            // 混合
            Mat one = Mat.Ones(alpha3.Size(), alpha3.Type());
            Mat invAlpha3 = one - alpha3;
            Mat resultF = fgBGRf.Mul(alpha3) + bgrF.Mul(invAlpha3);

            // 转回8位
            Mat result = new Mat();
            resultF.ConvertTo(result, MatType.CV_8UC3, 255.0);

            // 释放中间Mat
            foreach (var m in bgraChannels) m.Dispose();
            alpha.Dispose();
            alphaF.Dispose();
            alpha3.Dispose();
            fgBGR.Dispose();
            fgBGRf.Dispose();
            bgrF.Dispose();
            resultF.Dispose();

            return result;
        }
    }
}
