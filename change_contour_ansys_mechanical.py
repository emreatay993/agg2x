import clr
# reference WinForms and Drawing assemblies
clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")
# reference Ansys utilities
clr.AddReference("Ans.Utilities")

import System.Windows.Forms as WinForms
import System.Drawing as Drawing

# ——— MOVE THESE TO THE TOP ———
WinForms.Application.EnableVisualStyles()
# ————————————————————————

def GetMinMax():
    try:
        legendSettings = Ansys.Mechanical.Graphics.Tools.CurrentLegendSettings()
        noBands = legendSettings.NumberOfBands
        legendData = []
        for i in range(noBands):
            legendData.append( legendSettings.GetUpperBound(noBands - i - 1).Value )
        legendData.append( legendSettings.GetLowerBound(0).Value )
        return legendData, noBands
    except Exception, e:
        ExtAPI.Log.WriteError("Exception in GetMinMax(): " + str(e))

class ContourForm(WinForms.Form):
    def __init__(self):
        WinForms.Form.__init__(self)
        self.Text           = "Contour Editor"
        self.StartPosition  = WinForms.FormStartPosition.CenterScreen
        self.Font           = Drawing.Font("Segoe UI", 9)
        self.AutoSize       = True
        self.AutoSizeMode   = WinForms.AutoSizeMode.GrowAndShrink

        self.text_boxes = []
        self._build_ui()

    def _build_ui(self):
        legendData, noBands = GetMinMax()
        self.originalLegendData = list(legendData)
        res_unit = Tree.FirstActiveObject.Maximum.Unit

        panel = WinForms.FlowLayoutPanel()
        panel.FlowDirection = WinForms.FlowDirection.TopDown
        panel.WrapContents   = False
        panel.AutoSize       = True
        panel.AutoSizeMode   = WinForms.AutoSizeMode.GrowAndShrink
        panel.Dock           = WinForms.DockStyle.Fill
        self.Controls.Add(panel)

        for idx, val in enumerate(legendData):
            lbl = WinForms.Label()
            lbl.Text     = "Value {0} [{1}]".format(idx+1, res_unit)
            lbl.AutoSize = True
            panel.Controls.Add(lbl)

            tb = WinForms.TextBox()
            tb.Text  = str(val)
            tb.Width = 100
            panel.Controls.Add(tb)
            self.text_boxes.append(tb)

        btn = WinForms.Button()
        btn.Text     = "Update Contour"
        btn.AutoSize = True
        btn.Click   += self._on_click
        panel.Controls.Add(btn)

    def _on_click(self, sender, args):
        res_unit = Tree.FirstActiveObject.Maximum.Unit
        legendSettings = Ansys.Mechanical.Graphics.Tools.CurrentLegendSettings()
        noBands = legendSettings.NumberOfBands

        inputs = []
        for tb in self.text_boxes:
            txt = tb.Text.replace(',', '.')
            try:
                inputs.append(float(txt))
            except:
                inputs.append(0.0)

        changed = None
        for i in range(len(inputs)):
            if abs(inputs[i] - self.originalLegendData[i]) > 1e-6:
                changed = i
                break

        if changed is not None and changed < len(inputs) - 1:
            bottom = self.originalLegendData[-1]
            top    = inputs[changed]
            steps  = (len(inputs)-1) - changed
            delta  = (top - bottom) / float(steps)

            new_vals = []
            for i in range(len(inputs)):
                if i <= changed:
                    new_vals.append(inputs[i])
                elif i < len(inputs)-1:
                    new_vals.append(top - delta*(i - changed))
                else:
                    new_vals.append(bottom)

            for i, tb in enumerate(self.text_boxes):
                tb.Text = str(new_vals[i])
            self.originalLegendData = list(new_vals)
            inputs = new_vals

        for i, val in enumerate(inputs):
            q = Quantity(val, res_unit)
            try:
                legendSettings.SetUpperBound(noBands - i - 1, q)
            except:
                legendSettings.SetLowerBound(0, q)

def ChangeContours():
    try:
        form = ContourForm()
        form.ShowDialog()
    except Exception, e:
        ExtAPI.Log.WriteError("Exception in ChangeContours(): " + str(e))

# launch
ChangeContours()
