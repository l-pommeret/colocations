# Détection de Collocations en Grec Ancien

Ce projet propose un outil simple pour extraire et identifier les collocations (associations fréquentes de mots) dans un corpus de textes en Grec Ancien.

Il utilise une approche statistique basée sur le **PMI (Pointwise Mutual Information)** couplé à une **lemmatisation** via `stanza` (Stanford NLP), ce qui permet de repérer des expressions figées ou des formules poétiques indépendamment des conjugaisons ou déclinaisons.

## Fonctionnalités

*   **Lemmatisation automatique** : Transforme les mots fléchis en leur forme dictionnaire (ex: *ἔλαβε* -> *λαμβάνω*).
*   **Fenêtre glissante** : Repère les associations de mots même s'ils ne sont pas strictement contigus (fenêtre de 5 mots).
*   **Ranking intelligent** : Utilise le score PMI pour favoriser les associations significatives (formules rares) plutôt que simplement fréquentes (comme "et le").

## Installation

1.  **Pré-requis** : Python 3.8+
2.  **Installer les dépendances** :

```bash
pip install stanza
```

## Utilisation

1.  Placez votre corpus (fichier texte) dans le dossier. Par défaut, le script cherche `homer.iliad.tess`.
2.  Lancez le script d'analyse :

```bash
python3 analyze_collocations.py
```

Lors de la première exécution, le script téléchargera automatiquement le modèle de langue Grec Ancien (~200-300 Mo).

## Méthodologie

Le script procède en 3 étapes :

1.  **Nettoyage & Lemmatisation** : Le texte est nettoyé (suppression des balises XML/Tesserae) et passé dans le pipeline NLP `stanza` pour extraire les lemmes.
2.  **Comptage des Co-occurrences** : On compte combien de fois deux lemmes apparaissent ensemble dans une fenêtre de 5 mots.
3.  **Calcul du PMI** :
    $$ PMI(x, y) = \log_2 \frac{P(x, y)}{P(x)P(y)} $$
    Ce calcul permet de mesurer si deux mots apparaissent ensemble plus souvent que le hasard ne le prédirait.

## Exemple de Résultats (Iliade)

Le script permet de retrouver des formules homériques célèbres :
*   *('ῥοδοδάκτυλος', 'εὐωός')* -> L'Aurore aux doigts de roses.
*   *('Δεῖμός', 'Φόβος')* -> La Crainte et la Terreur.
*   *('τρέρων', 'πέλειος')* -> La colombe craintive.
