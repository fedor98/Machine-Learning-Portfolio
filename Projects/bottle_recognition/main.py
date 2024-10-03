import signal
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, DirCreatedEvent, FileCreatedEvent
from typing import Union
import pyfiglet

from cv import run as cv_run

file_format = "jpeg"
image_fragction = 4

class FileCreationHandler(FileSystemEventHandler):
    def __init__(self, files_to_wait_for, callback):
        self.files_to_wait_for = set(files_to_wait_for)
        self.files_found = set()
        self.callback = callback

    def on_created(self, event: Union[DirCreatedEvent, FileCreatedEvent]):
        if event.src_path.endswith(tuple(self.files_to_wait_for)):

            # extract the file name from the whole path
            file_name = event.src_path.split('/')[-1]       # 'event.src_path': represents the absolute path of the added item
            self.files_found.add(file_name)

            if file_name == f"L.{file_format}":
                print("CAMERA L: done")

            elif file_name == f"R.{file_format}":
                print("CAMERA R: done")

            # Check if all files are found
            if self.files_to_wait_for == self.files_found:
                print("All required files detected.\n")
                self.callback()
                observer.stop()

def watch_for_files():
    print("start watching\n")
    global observer
    path = "./photos"  # Directory to watch
    files_to_wait_for = [f"L.{file_format}", f"R.{file_format}"]

    event_handler = FileCreationHandler(files_to_wait_for, execute_code)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()  # method is used to block the main thread until the observer thread terminates

def signal_handler(sig, frame):
    print(pyfiglet.figlet_format("stopping ..."))
    observer.stop()
    observer.join()
    exit(0)

def execute_code():
    print("Computer Vision starts")
    image_paths = [f"./photos/L.{file_format}", f"./photos/R.{file_format}"]
    try:
        bottle_box_dict = cv_run(image_paths, image_fragction)
        print("\nQR codes processed successfully.\n")

        print(bottle_box_dict)
    except Exception as e:
        print(f"{e}")
        
        # Exit the function after an error
        return None, None


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print(pyfiglet.figlet_format("ROBOTTLE"))
    while True:
        watch_for_files()
        print("Restarting the file watch process.")

