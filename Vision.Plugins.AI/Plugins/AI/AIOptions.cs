using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Vision.Plugins.AI
{
    public class AIOptions
    {
        public static SessionOptions SessionOptions { get; } = new SessionOptions();
    }
}
