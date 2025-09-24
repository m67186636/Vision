using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI
{
    internal class DataUtils
    {
        public static Mat BCHW_to_BCWH(Mat bchw)
        {
            if (bchw.Dims != 4)
                throw new ArgumentException("Input Mat must be 4D");

            int B = bchw.Size(0);
            int C = bchw.Size(1);
            int H = bchw.Size(2);
            int W = bchw.Size(3);

            Mat bcwh = new Mat(new int[] { B, C, W, H }, bchw.Type());

            for (int b = 0; b < B; b++)
                for (int c = 0; c < C; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                        {
                            // BCHW: [b, c, h, w] -> BCWH: [b, c, w, h]
                            if (bchw.Type() == MatType.CV_32F)
                            {
                                float val = bchw.At<float>(b, c, h, w);
                                bcwh.Set([b, c, w, h], val);
                            }
                            else if (bchw.Type() == MatType.CV_64F)
                            {
                                double val = bchw.At<double>(b, c, h, w);
                                bcwh.Set([b, c, w, h], val);
                            }
                            else
                            {
                                throw new NotSupportedException("Only CV_32F and CV_64F are supported");
                            }
                        }
            return bcwh;
        }
    }
}
