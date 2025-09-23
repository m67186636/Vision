using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Vision.Plugins.AI.Onnx;

namespace Vision
{
    public partial class MainForm : Form
    {
        public PictureBox Before { get; }
        public PictureBox After { get; }
        public MainForm()
        {
            InitializeComponent();
            (Before,After) =InitializeControls();
            DoubleBuffered = true;
            VideoCaptureHelper.Instence.OnFrameCaptured += OnFrameCaptured;
        }

        private void OnFrameCaptured(Bitmap bitmap1, Bitmap bitmap2)
        {
            BeginInvoke(() => { 
                Before.Image?.Dispose();
                Before.Image = bitmap1;
                After.Image?.Dispose();
                After.Image = bitmap2;
                
                var height = bitmap1.Height > bitmap2.Height ? bitmap1.Height : bitmap2.Height;
                if (height > Before.Parent.Parent.Height)
                {
                    var radio = (float)bitmap2.Height / bitmap2.Width;
                    height = Before.Parent.Parent.Height;
                    var width=(int)(height/radio);
                    Before.Width = width;
                    After.Width = width;
                }
                else
                {
                    Before.Width = bitmap1.Width;
                    After.Width = bitmap2.Width;
                }
                    Before.Parent.Height = height;
            });
        }

        private (PictureBox Before, PictureBox After) InitializeControls()
        {
            var panel = new LayoutPanel
            {
                Dock = DockStyle.Top
            };
            Controls.Add(panel);
            var before = new PictureBox
            {
                SizeMode = PictureBoxSizeMode.Zoom,
            };
            panel.Controls.Add(before);
            var after = new PictureBox
            {
                SizeMode = PictureBoxSizeMode.Zoom,
                BackColor = Color.Black,
            };
            panel.Controls.Add(after);
            return (before, after);
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            VideoCaptureHelper.Instence.SetProcessor<RobustVideoMattingProcessor>();
            Task.Run(VideoCaptureHelper.Instence.StartAsync);
        }
        protected override void OnClosed(EventArgs e)
        {
            VideoCaptureHelper.Instence.OnFrameCaptured -= OnFrameCaptured;
            base.OnClosed(e);
        }
    }
}
