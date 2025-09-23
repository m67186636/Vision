using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.Layout;

namespace Vision
{
    public partial class LayoutPanel : UserControl
    {
        public override LayoutEngine LayoutEngine { get; }
        public LayoutPanel() {
            LayoutEngine = new PanelLayoutEngine();
        }
    }
    public class PanelLayoutEngine : LayoutEngine
    {
        public override bool Layout(object container, LayoutEventArgs layoutEventArgs)
        {
            var parent = (Control)container;
            var height= parent.Height;
            var x = parent.Padding.Left;
            var y = parent.Padding.Top;
            height -= (parent.Padding.Top + parent.Padding.Bottom);
            var gap = 5;

            foreach (Control control in parent.Controls)
            {
                control.Location = new Point(x, y);
                control.Size = new Size(control.Size.Width, height);
                x += control.Size.Width;
                x += gap;
            }

            return false;
        }
    }
    public class PanelVerticalLayoutEngine : LayoutEngine
    {
        public override bool Layout(object container, LayoutEventArgs layoutEventArgs)
        {
            var parent = (Control)container;
            var width = parent.Width;
            var x = parent.Padding.Left;
            var y = parent.Padding.Top;
            width -= (parent.Padding.Left + parent.Padding.Right);
            var gap = 5;

            foreach (Control control in parent.Controls)
            {
                control.Location = new Point(x, y);
                control.Size = new Size(width, control.Size.Height);
                y += control.Size.Height;
                y += gap;
            }

            return false;
        }
    }
}
