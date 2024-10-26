import tkinter as tk
from tkinter import filedialog, messagebox
import os
from time import sleep
import threading

app = tk.Tk()
app.title("Tracker")
app.geometry("960x720")

# Create a label widget
label = tk.Label(app, text="Choose files to analyze", font=("Arial", 14))
label.pack(pady=10)  # Add padding around the label


def on_browse_click():
    initial_directory = os.path.expanduser("~/Videos")
    file_paths = filedialog.askopenfilenames(
        title="Select a File",
        initialdir=initial_directory,
        filetypes=[("MP4 Files", "*.mp4")]
    )
    if len(file_paths) < 1:
        return

    if file_paths:
        thread = threading.Thread(target=process_files, args=(file_paths,))
        thread.start()
    

def process_files(paths):
    for path in paths:
        file_label = tk.Label(app, text=f'Processing: {path}', font=("Arial", 14), justify='left')
        file_label.pack(pady=10)
        analyze_file(path)
        file_label.destroy()


def analyze_file(path):
    print(path)
    sleep(1)


button = tk.Button(app, text="Browse", command=on_browse_click)
button.pack(pady=20)

# Start the application's main event loop
app.mainloop()
