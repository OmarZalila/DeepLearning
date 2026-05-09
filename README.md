# Show, Attend and Tell - PyTorch Flickr8k

Ce projet contient les modeles utilises pour comparer Flickr8k avec le papier :

- `soft` : Show, Attend and Tell avec soft attention.
- `hard` : Show, Attend and Tell avec hard attention stochastique.
- `nic` : baseline Google NIC, sans attention.
- `log_bilinear` : baseline Log-Bilinear-style, avec feature image globale.

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Place les fichiers comme ceci :

```text
data/raw/Flickr8k_Dataset/
data/raw/captions.txt
```

Pour Flickr30k, place les fichiers comme ceci :

```text
data/raw2/flickr30k_images/
data/raw2/captions.txt
```

Le loader accepte les deux formats de captions :

```text
image,caption
image_name,comment_number,comment
```

## Entrainer un modele

Le code fait maintenant le split par image unique. Les 5 captions d'une meme image restent ensemble dans train ou validation, ce qui evite de gonfler BLEU/METEOR.

```powershell
.\venv\Scripts\python.exe train.py --model soft --epochs 10
.\venv\Scripts\python.exe train.py --model hard --epochs 10
.\venv\Scripts\python.exe train.py --model nic --epochs 10
.\venv\Scripts\python.exe train.py --model log_bilinear --epochs 10
```

Pour entrainer sur Flickr30k sans remplacer les checkpoints Flickr8k :

```powershell
.\venv\Scripts\python.exe train.py --model nic --data-root data/raw2/flickr30k_images --captions-file data/raw2/captions.txt --checkpoint-dir outputs/checkpoints_flickr30k --epochs 10
.\venv\Scripts\python.exe train.py --model soft --data-root data/raw2/flickr30k_images --captions-file data/raw2/captions.txt --checkpoint-dir outputs/checkpoints_flickr30k --epochs 10
.\venv\Scripts\python.exe train.py --model hard --data-root data/raw2/flickr30k_images --captions-file data/raw2/captions.txt --checkpoint-dir outputs/checkpoints_flickr30k --epochs 10
.\venv\Scripts\python.exe train.py --model log_bilinear --data-root data/raw2/flickr30k_images --captions-file data/raw2/captions.txt --checkpoint-dir outputs/checkpoints_flickr30k --epochs 10
```

Pour entrainer les quatre modeles :

```powershell
.\venv\Scripts\python.exe train.py --model all --epochs 10
```

Les checkpoints sont sauvegardes separement :

```text
outputs/checkpoints/soft/best_checkpoint.pth
outputs/checkpoints/hard/best_checkpoint.pth
outputs/checkpoints/nic/best_checkpoint.pth
outputs/checkpoints/log_bilinear/best_checkpoint.pth
```

## Tester une image

```powershell
.\venv\Scripts\python.exe inference.py --image data/raw/Flickr8k_Dataset/12830823_87d2654e31.jpg --checkpoint outputs/checkpoints/nic/best_checkpoint.pth --visualize
```

L'evaluation affiche BLEU-1, BLEU-2, BLEU-3, BLEU-4 et METEOR quand NLTK a les ressources necessaires.
