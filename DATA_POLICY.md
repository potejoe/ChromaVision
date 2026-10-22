# Données et diffusion

Le dépôt public contient le code, les modèles couleur, les métriques et des
graphiques. Il ne contient aucune photographie source, masque d'annotation
du jeu, capture de visage ou sortie de notebook montrant les personnes.

Les 71 paires utilisées proviennent du sous-ensemble Pratheepan livré dans
le Lab 1 du cours. Les données et les annotations restent des contenus tiers,
indépendants du code. Les auteurs les mettent à disposition pour usage non
commercial : https://cs-chan.com/downloads_skin_dataset.html.

Attribution : W. R. Tan, C. S. Chan, P. Yogarajah et J. Condell,
*A Fusion Approach for Efficient Human Skin Detection*, IEEE Transactions on
Industrial Informatics 8(1), 138–147, 2012.
https://doi.org/10.1109/TII.2011.2172451.

Pour reproduire exactement l'expérience, utiliser le sous-ensemble de 71 paires
du TP et comparer les empreintes au manifeste. Les fichiers d'origine restent
dans `data/raw/`, ignoré par Git. Le modèle n'effectue pas de reconnaissance
d'identité ou d'attributs démographiques.
