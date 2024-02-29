import clr
clr.AddReference('System.Windows.Forms')
clr.AddReference('System.Drawing')

from System.Windows.Forms import (Application, Form, Label, TextBox, Button, 
                                  OpenFileDialog, DialogResult, FolderBrowserDialog, 
                                  Keys)
from System.Drawing import Size, Point

class CommandBatchForm(Form):
    def __init__(self):
        self.InitializeComponent()
    
    def InitializeComponent(self):
        # Set up the form
        self.Text = 'Define Commands in Mechanical for Each Timestep'
        self.Size = Size(500, 380)
        
        # Label for Grouping Folder Name
        self.labelGroupName = Label()
        self.labelGroupName.Text = 'Grouping Folder Name for commands:'
        self.labelGroupName.Location = Point(10, 20)
        self.labelGroupName.AutoSize = True
        
        # TextBox for Grouping Folder Name
        self.textBoxGroupName = TextBox()
        self.textBoxGroupName.Location = Point(10, 40)
        self.textBoxGroupName.Width = 300
        self.textBoxGroupName.KeyDown += self.OnKeyDown
        
        # Button to open file dialog
        self.buttonOpenFileDialog = Button()
        self.buttonOpenFileDialog.Text = 'Select a .txt or .inp file'
        self.buttonOpenFileDialog.Location = Point(10, 70)
        self.buttonOpenFileDialog.Click += self.OpenFileDialog_Click
        
        # TextBox for file path
        self.textBoxFilePath = TextBox()
        self.textBoxFilePath.Location = Point(10, 100)
        self.textBoxFilePath.Width = 300
        self.textBoxFilePath.ReadOnly = True
        
        # Multiline TextBox for file contents
        self.textBoxFileContents = TextBox()
        self.textBoxFileContents.Multiline = True
        self.textBoxFileContents.Location = Point(10, 130)
        self.textBoxFileContents.Size = Size(460, 100)
        self.textBoxFileContents.KeyDown += self.OnKeyDown
        
        # Label for the command snippets
        self.labelCommandSnippets = Label()
        self.labelCommandSnippets.Text = 'Name of the command snippets:'
        self.labelCommandSnippets.Location = Point(10, 240)
        self.labelCommandSnippets.AutoSize = True
        
        # TextBox for the command snippets
        self.textBoxCommandSnippets = TextBox()
        self.textBoxCommandSnippets.Location = Point(10, 260)
        self.textBoxCommandSnippets.Width = 300
        self.textBoxCommandSnippets.KeyDown += self.OnKeyDown
        
        # OK Button
        self.buttonOK = Button()
        self.buttonOK.Text = 'OK'
        self.buttonOK.Location = Point(10, 300)
        self.buttonOK.Location = Point(10, self.ClientSize.Height - 40)
        self.buttonOK.Width = self.ClientSize.Width - 20
        self.buttonOK.Click += self.ButtonOK_Click
        
        # Add controls to the form
        self.Controls.Add(self.labelGroupName)
        self.Controls.Add(self.textBoxGroupName)
        self.Controls.Add(self.buttonOpenFileDialog)
        self.Controls.Add(self.textBoxFilePath)
        self.Controls.Add(self.textBoxFileContents)
        self.Controls.Add(self.labelCommandSnippets)
        self.Controls.Add(self.textBoxCommandSnippets)
        self.Controls.Add(self.buttonOK)
        
    def OpenFileDialog_Click(self, sender, e):
        openFileDialog = OpenFileDialog()
        if openFileDialog.ShowDialog() == DialogResult.OK:
            self.textBoxFilePath.Text = openFileDialog.FileName
            self.LoadFile(openFileDialog.FileName)
    
    def LoadFile(self, path):
        with StreamReader(path) as reader:
            self.fileContent = reader.ReadToEnd()
            self.textBoxFileContents.Text = self.fileContent
    
    def ButtonOK_Click(self, sender, e):
        # Store user inputs in variables
        self.groupName = self.textBoxGroupName.Text
        self.filePath = self.textBoxFilePath.Text
        self.commandSnippets = self.textBoxCommandSnippets.Text
        # Save the content of the selected text file to a variable as a multicolumn string
        self.fileContent = self.textBoxFileContents.Text
        # Close the form to end the program
        self.Close()
    
    def OnKeyDown(self, sender, e):
        if e.KeyCode == Keys.Enter:
            self.ButtonOK_Click(sender, e)

# Create and run the form
form = CommandBatchForm()
Application.Run(form)
