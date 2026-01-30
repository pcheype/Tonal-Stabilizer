import os
from tkinter import Tk, filedialog
from frameloader import FrameLoader
from tonalprocessor import TonalProcessor
from formator import Formator

def main_pipeline():

    root = Tk()
    root.withdraw()
    img_folder = filedialog.askdirectory(title="Choisir le dossier contenant les images")
    if not img_folder:
        print("Aucun dossier sélectionné. Fin du programme.")
        return

    repo_folder = os.getcwd()  # ou mettre le chemin vers ton repo
    result_folder = os.path.join(repo_folder, "resultats")
    os.makedirs(result_folder, exist_ok=True)

    video_orig_path = os.path.join(result_folder, "video_originale.mp4")
    fl = FrameLoader(img_folder)
    fl.folder2video(video_orig_path)
    print(f"Vidéo originale générée -> {video_orig_path}")

    video_stab_path = os.path.join(result_folder, "video_stabilisee.mp4")
    tp = TonalProcessor(video_orig_path, video_stab_path)
    tp.stabilize()
    print(f"Vidéo stabilisée générée -> {video_stab_path}")

    video_comp_path = os.path.join(result_folder, "video_comparaison.mp4")
    fm = Formator()
    fm.compare_videos_diagonal(video_orig_path, video_stab_path, output_path=video_comp_path)
    print(f"Vidéo de comparaison générée -> {video_comp_path}")

    print("Pipeline terminée ✅ Toutes les vidéos sont dans le dossier 'resultats/'")


if __name__ == "__main__":
    main_pipeline()
