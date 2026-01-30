import cv2
import os
from tkinter import Tk, filedialog


class FrameLoader:

    def __init__(self, path=None):

        if path is None:
            self.path = self.ask_folder()
        else:
            self.path = path
    

    def ask_folder(self):

        root = Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Choisir un dossier d'images")

        if not folder:
            raise ValueError("Aucun dossier selectionné")
        
        print(f"Video choisie : {folder}")

        return folder


    def load_images(self):

        files = [f for f in os.listdir(self.path) 
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
        
        frames = []
        for f in files:
            img_path = os.path.join(self.path,f)
            img = cv2.imread(img_path)
            frames.append(img)

        return frames
    

    def create_video(self,frames,out_path,fps=30):

        h,w = frames[0].shape[:2]
        writer = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),fps, (w,h))
        for frame in frames:
            writer.write(frame)

        writer.release()

        print(f"Video genérée : {out_path}")

    def ask_output_folder(self):
        root = Tk()
        root.withdraw()

        folder = filedialog.askdirectory(title="Choisir le dossier de sortie")

        if not folder:
            raise ValueError("Aucun dossier de sortie sélectionné.")

        print(f"Dossier de sortie : {folder}")
        return folder


    def folder2video(self, out_path, fps=30):
        #if out_path is None:
        #    out_folder = self.ask_output_folder()
        #   out_path = os.path.join(out_folder, "output.mp4")

        frames = self.load_images()
        self.create_video(frames, out_path, fps)


if __name__ == "__main__":

    fl = FrameLoader()
    fl.folder2video("out.mp4",fps=30)