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
            VideoCaptureHelper.Instance.OnFrameCaptured += OnFrameCaptured;
            VideoFileHelper.Instance.OnFrameCaptured += OnFrameCaptured;
        }

        private void OnFrameCaptured(Bitmap bitmap)
        {
            BeginInvoke(() => {
                Before.Image?.Dispose();
                Before.Image = bitmap;
                After.Image?.Dispose();
                var height = bitmap.Height;
                if (height > Before.Parent.Parent.Height)
                {
                    var radio = (float)bitmap.Height / bitmap.Width;
                    height = Before.Parent.Parent.Height;
                    var width = (int)(height / radio);
                    Before.Width = width;
                    After.Width = width;
                }
                else
                {
                    Before.Width = bitmap.Width;
                    After.Width = bitmap.Width;
                }
                Before.Parent.Height = height;
            });
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
                BackColor = Color.DarkGreen,
            };
            panel.Controls.Add(after);
            return (before, after);
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            //VideoCaptureHelper.Instance.SetProcessor<RobustVideoMattingProcessor>();
            //Task.Run(VideoCaptureHelper.Instance.StartAsync);
            Task.Run(VideoFileHelper.Instance.StartAsync);
        }
        protected override void OnClosed(EventArgs e)
        {
            VideoCaptureHelper.Instance.OnFrameCaptured -= OnFrameCaptured;
            VideoFileHelper.Instance.OnFrameCaptured -= OnFrameCaptured;
            base.OnClosed(e);
        }
    }
}
