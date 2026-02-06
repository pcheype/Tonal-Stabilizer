import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QLabel, QHBoxLayout, QGroupBox, QPushButton, 
                               QProgressBar, QGridLayout, QSizePolicy, QDoubleSpinBox, QFileDialog, QMessageBox)
from PySide6.QtGui import QFont

from PySide6.QtCore import QTimer

from frameloader import VideoManager
from tonalprocessor import TonalProcessor
from formator import FormatorProcessor
import os, platform, subprocess

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.video_manager = VideoManager(parent=self)

        self.setWindowTitle("Tonal Stabilizer")
        self.resize(900, 200)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # --- Etape 1 ---
        etape1_box = QGroupBox("Etape 1 : Importation")
        etape1_layout = QVBoxLayout()
        etape1_layout.setSpacing(10)
        
        self.load_video = QPushButton("Choisir une vidéo (mp4)")
        self.load_imgs = QPushButton("Choisir une vidéo (dossier images)")


        self.set_button_style(self.load_video)
        self.set_button_style(self.load_imgs)

        
        self.progress_load = QProgressBar()
        
        self.visu_video_1 = QPushButton("Visualiser votre vidéo")
        
        self.visu_video_1.setEnabled(False)
        self.set_button_style(self.visu_video_1)
        
        etape1_layout.addWidget(self.load_video)
        etape1_layout.addWidget(self.load_imgs)
        etape1_layout.addWidget(self.progress_load)
        etape1_layout.addWidget(self.visu_video_1)
        etape1_box.setLayout(etape1_layout)

        # Dans ton __init__ de MainWindow :
        self.load_video.clicked.connect(self.on_load_video_clicked)
        self.load_imgs.clicked.connect(self.on_load_imgs_clicked)
        self.visu_video_1.clicked.connect(self.on_visu_video_clicked)

        # Initialise aussi la variable de suivi
        self.current_video_path = None

        # --- Etape 2 ---
        etape2_box = QGroupBox("Etape 2 : Processing")
        etape2_layout = QGridLayout()
        etape2_layout.setSpacing(10)
        
        # Configuration des QDoubleSpinBox
        
        # Lambda
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 10.0)   # Min 0, Max 10
        self.lambda_spin.setSingleStep(0.1)    # Incrément de 0.1
        self.lambda_spin.setValue(0.9)         # Valeur par défaut
        self.lambda_spin.setDecimals(2)        # 2 chiffres après la virgule

        # Omega
        self.omega_spin = QDoubleSpinBox()
        self.omega_spin.setRange(0.0, 1.0)     # Souvent entre 0 et 1 pour un ratio
        self.omega_spin.setSingleStep(0.05)
        self.omega_spin.setValue(0.7)
        self.omega_spin.setDecimals(2)
        
        # Sigma
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(0.1)
        self.sigma_spin.setDecimals(2)
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_processing)
        self.set_button_style(self.start_button)
        
        self.progress_processing = QProgressBar()
        
        etape2_layout.addWidget(QLabel("Facteur de régularisation Ⲗ = "), 0, 0)
        etape2_layout.addWidget(self.lambda_spin, 0, 1)
        
        etape2_layout.addWidget(QLabel("Paramètre de recouvrement ω ="), 1, 0)
        etape2_layout.addWidget(self.omega_spin, 1, 1)
        
        etape2_layout.addWidget(QLabel("Sigma σ ="), 2, 0)
        etape2_layout.addWidget(self.sigma_spin, 2, 1)

        etape2_layout.addWidget(self.start_button, 3, 0, 1, 2)
        etape2_layout.addWidget(self.progress_processing, 4, 0, 1, 2)
        
        etape2_layout.setRowStretch(3, 1)
        
        etape2_box.setLayout(etape2_layout)

        # --- Etape 3 ---
        etape3_box = QGroupBox("Etape 3 : Exportation")
        etape3_layout = QVBoxLayout()
        etape3_layout.setSpacing(10)
        
        self.download_button = QPushButton("Télécharger votre vidéo")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_video)
        self.set_button_style(self.download_button)
        
        self.visu_processvideo = QPushButton("Visualiser résultat")
        self.visu_processvideo.setEnabled(False)
        self.visu_processvideo.clicked.connect(self.on_visu_stabvideo_clicked)
        self.set_button_style(self.visu_processvideo)

        
        self.do_comp = QPushButton("Effectuer une comparaison")
        self.do_comp.setEnabled(False)
        self.do_comp.clicked.connect(self.on_compare_clicked)
        self.set_button_style(self.do_comp)

        self.downloadcomp_button = QPushButton("Télécharger et Visualiser la comparaison")
        self.downloadcomp_button.setEnabled(False)
        self.downloadcomp_button.clicked.connect(self.downvisu)
        self.set_button_style(self.downloadcomp_button)

        self.progress_comp = QProgressBar()
        
        etape3_layout.addWidget(self.download_button)
        etape3_layout.addWidget(self.visu_processvideo)

        etape3_layout.addWidget(self.do_comp)
        etape3_layout.addWidget(self.progress_comp)
        etape3_layout.addWidget(self.downloadcomp_button)

        etape3_box.setLayout(etape3_layout)

        # --- Container ---
        layout.addWidget(etape1_box)
        layout.addWidget(etape2_box)
        layout.addWidget(etape3_box)

        self.stab_path = None
        self.comb_path = None

    def set_button_style(self, button):
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        button.setMinimumHeight(40)
        font = QFont()
        font.setBold(True)
        button.setFont(font)

    def on_load_video_clicked(self):
        """Action pour le bouton 'Choisir une vidéo (mp4)'"""
        video_path = self.video_manager.get_path_from_dialog("video")
        if video_path:
            self.start_style_progress() # On lance l'animation
            save_path = self.video_manager.get_path_from_dialog("save")
            if save_path:
                # On traite la vidéo (conversion/copie en mp4v)
                success, msg = self.video_manager.process_existing_video(video_path, save_path)
                if success:
                    self.start_style_progress() # On lance l'animation
                    self.current_video_path = save_path # On stocke le chemin pour la visu
                    self.load_imgs.setEnabled(False)
                    self.visu_video_1.setEnabled(True)
                    self.start_button.setEnabled(True)

    def on_load_imgs_clicked(self):
        """Action pour le bouton 'Choisir une vidéo (dossier images)'"""
        folder_path = self.video_manager.get_path_from_dialog("folder")
        if folder_path:
            save_path = self.video_manager.get_path_from_dialog("save")
            if save_path:
                # Création de la vidéo à partir des images
                success, msg = self.video_manager.images_to_video(folder_path, save_path)
                if success:
                    self.current_video_path = save_path # On stocke le chemin pour la visu
                    self.start_style_progress() # On lance l'animation
                    self.load_video.setEnabled(False)
                    self.visu_video_1.setEnabled(True)
                    self.start_button.setEnabled(True)

                else:
                    QMessageBox.warning(self, "Erreur", f"Échec de la génération : {msg}")

    def on_visu_video_clicked(self):
        """Action pour le bouton 'Visualiser votre vidéo'"""
        # Vérifie si une vidéo a été créée ou chargée durant la session
        video_to_play = getattr(self, 'current_video_path', None)
        
        if video_to_play and os.path.exists(video_to_play):
            # Utilise le lecteur par défaut du système (Windows/Mac/Linux)
            if platform.system() == 'Windows':
                os.startfile(video_to_play)
            elif platform.system() == 'Darwin': # macOS
                subprocess.call(['open', video_to_play])
            else: # Linux
                subprocess.call(['xdg-open', video_to_play])
        else:
            QMessageBox.warning(self, "Attention", "Aucune vidéo n'est disponible pour la visualisation.")
  
    def start_style_progress(self):
        """Lance une progression fluide de 0 à 100% en 2 secondes."""
        self.progress_load.setValue(0)
        self.timer_step = 0
        self.total_steps = 20  # On va faire 50 mises à jour
        
        # 2000ms / 50 étapes = 40ms par étape
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(40) 

    def update_progress(self):
        self.timer_step += 1
        percentage = int((self.timer_step / self.total_steps) * 100)
        self.progress_load.setValue(percentage)
        
        if self.timer_step >= self.total_steps:
            self.progress_timer.stop()

    def start_processing(self):
        video_input = self.current_video_path
        lambda0 = self.lambda_spin.value()
        sigma = self.sigma_spin.value()
        omega = self.omega_spin.value()
        # On crée l'instance
        self.processor = TonalProcessor(video_input,120,sigma,omega,lambda0)
        self.processor.progress_update.connect(self.progress_processing.setValue)
        self.processor.finished_signal.connect(self.on_processing_finished)
        self.processor.start()
        self.download_button.setEnabled(True)
        

    def on_processing_finished(self, success, message):
        if success:
            self.download_button.setEnabled(True) # On active le téléchargement
            QMessageBox.information(self, "Succès", "Calcul terminé ! Vous pouvez maintenant télécharger la vidéo.")
        else:
            QMessageBox.critical(self, "Erreur", message)

    def download_video(self):
        # Demander à l'utilisateur où enregistrer
        save_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la vidéo", "stabilized_video.mp4", "Video (*.mp4)")
        self.stab_path = save_path
        
        if save_path:
            success, msg = self.processor.save_video(save_path)
            if success:
                QMessageBox.information(self, "Enregistré", f"Fichier sauvegardé : {msg}")
                self.do_comp.setEnabled(True)
                self.visu_processvideo.setEnabled(True)
            else:
                QMessageBox.critical(self, "Erreur", msg)

    def on_visu_stabvideo_clicked(self):
        """Action pour le bouton 'Visualiser votre vidéo'"""
        # Vérifie si une vidéo a été créée ou chargée durant la session
        video_to_play = self.stab_path
        
        if video_to_play and os.path.exists(video_to_play):
            # Utilise le lecteur par défaut du système (Windows/Mac/Linux)
            if platform.system() == 'Windows':
                os.startfile(video_to_play)
            elif platform.system() == 'Darwin': # macOS
                subprocess.call(['open', video_to_play])
            else: # Linux
                subprocess.call(['xdg-open', video_to_play])
        else:
            QMessageBox.warning(self, "Attention", "Aucune vidéo n'est disponible pour la visualisation.")
  

    def on_compare_clicked(self):
        self.compare_worker = FormatorProcessor(self.current_video_path,self.stab_path)
        self.compare_worker.progress_update.connect(self.progress_comp.setValue)
        self.compare_worker.finished_signal.connect(self.on_compare_finished)
        self.compare_worker.start()

    def on_compare_finished(self, success, message):
        if success:
            QMessageBox.information(self, "Calcul terminé", "La comparaison est prête. Cliquez sur Télécharger.")
            self.downloadcomp_button.setEnabled(True) # On active le bouton de sauvegarde
        else:
            QMessageBox.critical(self, "Erreur", message)

    def downvisu(self):

        save_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la vidéo", "stabilized_video.mp4", "Video (*.mp4)")
        self.comp_path = save_path
        
        if save_path:
            success, msg = self.compare_worker.save_comparison_video(save_path)
            if success:
                QMessageBox.information(self, "Enregistré", f"Fichier sauvegardé : {msg}")
            else:
                QMessageBox.critical(self, "Erreur", msg)

        """Action pour le bouton 'Visualiser votre vidéo'"""
        # Vérifie si une vidéo a été créée ou chargée durant la session
        video_to_play = self.comp_path
        
        if video_to_play and os.path.exists(video_to_play):
            # Utilise le lecteur par défaut du système (Windows/Mac/Linux)
            if platform.system() == 'Windows':
                os.startfile(video_to_play)
            elif platform.system() == 'Darwin': # macOS
                subprocess.call(['open', video_to_play])
            else: # Linux
                subprocess.call(['xdg-open', video_to_play])
        else:
            QMessageBox.warning(self, "Attention", "Aucune vidéo n'est disponible pour la visualisation.")



def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()