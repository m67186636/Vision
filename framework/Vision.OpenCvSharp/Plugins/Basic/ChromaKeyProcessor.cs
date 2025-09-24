using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.Basic
{
    public class ChromaKeyProcessor : IProcessor
    {
        // 复用Mat和数组，减少分配
        private readonly Mat hsv = new();
        private readonly Mat mainColorHSVMat = new();
        private readonly Mat tmp = new(1, 1, MatType.CV_8UC3);
        private readonly Mat mask = new();
        private readonly Mat edge = new();
        private readonly Mat edgeDilated = new();
        private readonly Mat kernel = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(5, 5));
        private readonly Mat smoothMask = new();
        private readonly Mat blurred = new();
        private readonly Mat alpha = new();
        private readonly Mat[] bgr = new Mat[3];
        private readonly Mat[] bgra = [new(),new(),new(),new()];
        private readonly Mat alphaF = new();
        public Task ExecuteAsync(Mat image)
        {
            var colorTolerance = 10;
            // 1. 计算主色（BGR）
            var mainColor = ChromaKeyUtils.DetectMainColor(image);
            // 2. 转为HSV，主色也要转为HSV
            Cv2.CvtColor(image, hsv, ColorConversionCodes.BGR2HSV);
            tmp.SetTo(mainColor);
            Cv2.CvtColor(tmp, mainColorHSVMat, ColorConversionCodes.BGR2HSV);
            Vec3b mainColorHSV = mainColorHSVMat.At<Vec3b>(0, 0);

            // 3. 设定抠像区间
            int h = mainColorHSV.Item0, s = mainColorHSV.Item1, v = mainColorHSV.Item2;
            var lower = new Scalar(Math.Max(h - colorTolerance, 0), 100, 100);
            var upper = new Scalar(Math.Min(h + colorTolerance, 255), 255, 255);

            // 4. 生成mask，主色区域=255
            Cv2.InRange(hsv, lower, upper, mask);


            Cv2.Canny(mask, edge, 100, 200);
            Cv2.Dilate(edge, edgeDilated, kernel);
            mask.CopyTo(smoothMask);
            Cv2.GaussianBlur(mask, blurred, new Size(5, 5), 0);
            blurred.CopyTo(smoothMask, edgeDilated);
            smoothMask.CopyTo(mask);
            //mask.SetTo(smoothMask);

            Cv2.GaussianBlur(mask, mask, new Size(15, 15), 0); // 15x15核，边缘更柔和

            // 5. 优化mask
            Cv2.MedianBlur(mask, mask, 5);

            // 6. 生成alpha通道（主色区域变透明）
            Cv2.BitwiseNot(mask, alpha);

            // 7. 合成BGRA
            Cv2.Split(image,out var bgr);

            // alpha归一化为float
            alpha.ConvertTo(alphaF, MatType.CV_32FC1, 1.0 / 255);
            // bgr转float
            for (int i = 0; i < 3; i++)
                bgr[i].ConvertTo(bgr[i], MatType.CV_32FC1, 1.0 / 255);

            // 用OpenCV矢量化方式乘以alpha
            for (int i = 0; i < 3; i++)
                Cv2.Multiply(bgr[i], alphaF, bgr[i]);

            // 合并为BGRA
            for (int i = 0; i < 3; i++)
                bgr[i].ConvertTo(bgra[i], MatType.CV_8UC1, 255.0);
            bgra[3] = alpha.Clone();
            Cv2.Merge(bgra, image);
            return Task.CompletedTask;
            for (int y = 0; y < image.Rows; y++)
            {
                for (int x = 0; x < image.Cols; x++)
                {
                    Vec4b color = image.At<Vec4b>(y, x);
                    byte a = color[3];
                    color[0] = (byte)(color[0] * a / 255); // B
                    color[1] = (byte)(color[1] * a / 255); // G
                    color[2] = (byte)(color[2] * a / 255); // R
                    image.Set(y, x, color);
                }
            }
            // 释放
            hsv.Dispose(); mainColorHSVMat.Dispose(); tmp.Dispose(); mask.Dispose(); alpha.Dispose();
            foreach (var c in bgr) c.Dispose();
            return Task.CompletedTask;

        }

        private Point FindCenter(Mat alpha)
        {
            var pointsMat = new Mat();
            Cv2.FindNonZero(alpha,pointsMat);
            var pts = new Point[pointsMat.Rows];
            for (int i = 0; i < pointsMat.Rows; i++)
                pts[i] = pointsMat.Get<Point>(i);
            return new Point((int)pts.Average(p => p.X), (int)pts.Average(p => p.Y));
        }
    }
    public class ChromaKeyUtils
    {
        /// <summary>
        /// 自动检测主色（返回BGR）
        /// </summary>
        /// <param name="src">输入图像Mat</param>
        /// <param name="k">聚类中心数量，主色数量</param>
        /// <returns>主色的BGR Scalar</returns>
        public static Scalar DetectMainColor(Mat src, int k = 3)
        {
            // 缩小图像加速
            Mat small = new Mat();
            Cv2.Resize(src, small, new Size(100, 100), 0, 0, InterpolationFlags.Area);

            // 展平成N*3
            Mat reshaped = small.Reshape(1, rows: small.Rows * small.Cols);
            reshaped.ConvertTo(reshaped, MatType.CV_32F);

            // KMeans 聚类
            TermCriteria criteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 10, 1.0);
            Mat labels = new Mat();
            Mat centers = new Mat();
            Cv2.Kmeans(reshaped, k, labels, criteria, 3, KMeansFlags.PpCenters, centers);

            // 统计每类像素数
            int[] counts = new int[k];
            for (int i = 0; i < labels.Rows; i++)
                counts[labels.Get<int>(i, 0)]++;

            // 找到最大类
            int mainIdx = Array.IndexOf(counts, counts.Max());

            // 得到主色（BGR）
            float b = centers.At<float>(mainIdx, 0);
            float g = centers.At<float>(mainIdx, 1);
            float r = centers.At<float>(mainIdx, 2);

            small.Dispose();
            reshaped.Dispose();
            labels.Dispose();
            centers.Dispose();

            return new Scalar(b, g, r); // B, G, R
        }

        public static Mat CreateColorImage(Scalar color, int width = 400, int height = 300)
        {
            return new Mat(height, width, MatType.CV_8UC3, color);
        }
    }
}
