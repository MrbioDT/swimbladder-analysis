import Tkinter,tkFileDialog
import tkMessageBox
import sys
import pdb

def pickfile(path='.'): #add filetypes
    root = Tkinter.Tk()
    root.withdraw()
    f = tkFileDialog.askopenfilename(parent=root,title='Choose a file')
    # returned f is a tuple
    # The askopenfile function to creates an file dialog object. The extensions are shown in the bottom of the form (Files of type). The code below will simply show the dialog and return the filename.
    if f:
        root.destroy()
        del root
        return f
    else:
        print "No file picked, exiting!"
        root.destroy()
        del root
        sys.exit()

def saveasfile(path='.', filetypes = [], defaultextension=''): #add filetypes
    root = Tkinter.Tk()
    root.withdraw()
    f = tkFileDialog.asksaveasfilename(parent=root,title='Choose a filepath to save as',filetypes = filetypes,defaultextension=defaultextension)
    # The asksaveasfilename function prompts the user with a save file dialog.
    if f:
        root.destroy()
        del root
        return f
    else:
        print "No file picked, exiting!"
        root.destroy()
        del root
        sys.exit()
        
def pickfiles(path='.', filetypes = [], defaultextension=''): 
    root = Tkinter.Tk()
    # class Tkinter.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
    root.withdraw()
    # get rid of window of tkinter temporarily
    # root.deiconify(), If you want to make the window visible again, call the deiconify (or wm_deiconify) method.
    # root.destroy(), Once you are done with the dialog(s) you are creating, you can destroy the root window along with all other tkinter widgets with the destroy method
    f = tkFileDialog.askopenfilenames(parent=root,title='Choose a file',filetypes = filetypes) #tag
    if f:
        f=root.tk.splitlist(f)
        root.destroy()
        del root
        return f
    else:
        print "No file picked, exiting!"
        root.destroy()
        del root
        sys.exit()
        

def pickdir(path='.'):
    root = Tkinter.Tk()
    root.withdraw()
    dirname = tkFileDialog.askdirectory(parent=root,initialdir=".",title='Please select a directory')

    root.destroy()
    if len(dirname ) > 0:
        return dirname
    else:
        print "No directory picked, exiting!"
        sys.exit()

def askyesno(title = 'Display?',text = "Use interactive plotting?"):
    root = Tkinter.Tk()
    root.withdraw()
    tf = tkMessageBox.askyesno(title, text)   #tag
    root.destroy()
    return tf




    
##
### ======== "Save as" dialog:
##import Tkinter,tkFileDialog
##
##myFormats = [
##    ('Windows Bitmap','*.bmp'),
##    ('Portable Network Graphics','*.png'),
##    ('JPEG / JFIF','*.jpg'),
##    ('CompuServer GIF','*.gif'),
##    ]
##
##root = Tkinter.Tk()
##fileName = tkFileDialog.asksaveasfilename(parent=root,filetypes=myFormats ,title="Save the image as...")
##if len(fileName ) > 0:
##    print "Now saving under %s" % nomFichier

if __name__=='__main__':
##    f = pickfile()
##    d = pickdir()
##    s=saveasfile(filetypes = [("Shelve files",'*.shv'), ('All','*') ],defaultextension='.shv')
    fs = pickfiles(filetypes=[('AVI videos','*.avi'),('All files','*.*')])
    print fs
    print type(fs)
