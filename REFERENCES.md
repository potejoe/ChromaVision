# Références et lien aux cours

## Articles
1. W. R. Tan, C. S. Chan, P. Yogarajah, J. Condell (2012). A Fusion Approach for
   Efficient Human Skin Detection. IEEE Transactions on Industrial Informatics,
   8(1), 138–147. DOI : https://doi.org/10.1109/TII.2011.2172451.
   Version auteur : https://arxiv.org/abs/1410.3751.
2. M. J. Jones, J. M. Rehg (2002). Statistical Color Models with Application to
   Skin Detection. International Journal of Computer Vision 46, 81–96.
   https://doi.org/10.1023/A:1013200319198.
   Le projet reprend l'idée générale des histogrammes conditionnels, pas leur
   corpus d'un milliard de pixels ni leurs performances.
3. P. Kakumanu, S. Makrogiannis, N. Bourbakis (2007). A survey of skin-color
   modeling and detection methods. Pattern Recognition 40(3), 1106–1122.
   https://doi.org/10.1016/j.patcog.2006.06.010.
4. V. Vezhnevets, V. Sazonov, A. Andreeva (2003). A Survey on Pixel-Based Skin
   Color Detection Techniques. GraphiCon. Règle RGB telle que présentée dans
   votre TP Lab 1, sans prétendre répliquer l'ensemble des méthodes de l'article.

## Documentation primaire
- Pratheepan : https://cs-chan.com/downloads_skin_dataset.html.
- Conversions : https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html.
- Morphologie : https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html.

## Supports fournis, utilisés comme point de départ
- Eva Feillet, Computer Vision M2 AI, Université Paris-Saclay, automne 2025.
- Lecture 1.1 : démarche en vision et instructions du projet.
- Lecture 1.2 : RGB, HSV, Lab, représentation des couleurs.
- Lab 1 : règle RGB, histogrammes conditionnels et évaluation par annotations.
- Lab 2 : filtrage, bords, effets des prétraitements et temps de calcul.
- Lecture 4.1 : description d'image et représentations.
- Lecture 5 : segmentation et seuillage.
- CNN et ViT : approches possibles pour une extension future, non implémentées ici.
- Lecture 6 sur le mouvement : hors périmètre car le jeu contient des images fixes.

Apports du présent projet au-delà du TP : séparation stricte des images, gel
des modèles, grille de validation, étude contrôlée de perturbations, contrastes
appariés, bootstrap par image, morphologie, Gray World, profil mémoire et
expériences de réduction du nombre d'images d'apprentissage.
