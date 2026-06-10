# 🤖 Comment fonctionnent les chatbots ?

Application Streamlit pédagogique pour illustrer les principes fondamentaux des LLMs et chatbots d'IA, conçue pour un atelier lycéens. Chaque mode décompose une brique de plus en plus complexe, de la complétion brute jusqu'à la boucle agentique.

---

## Démonstrations

| Mode | Ce qu'on montre |
|---|---|
| **① Complétion de texte** | Le modèle prédit les tokens suivants un par un, en streaming |
| **② Dialogue** | Un chatbot n'est qu'une complétion d'historique de conversation passé en texte brut |
| **③ Raisonnement** | Un prompt force le modèle à écrire ses étapes de réflexion avant de répondre |
| **④ Agent avec outils** | Raisonnement + appel d'outil (calculatrice, météo) + boucle |

---

## Installation

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

### Dépendances principales

```
streamlit
openai
python-dotenv
httpx
truststore
```

---

## Configuration

L'app supporte deux modes de configuration, par ordre de priorité :

### 1. Variables d'environnement (recommandé)

Créez un fichier `.env` à la racine :

```env
DS_LLM_API_URL=https://your-api-endpoint/v1
DS_LLM_API_KEY=your-api-key
```

### 2. Saisie manuelle dans l'interface

Si les variables d'environnement ne sont pas définies, l'app affiche un formulaire dans la barre latérale pour saisir l'URL et la clé API directement. Utile pour un usage dans différents contextes (réseau interne, cloud, OpenAI, Mistral, Infomaniak...).

> Les valeurs saisies manuellement sont stockées en `session_state` — elles ne persistent pas au redémarrage de l'app.

---

## Lancement

```bash
streamlit run app.py
```

---

## Modèles supportés

L'app filtre les modèles disponibles sur l'endpoint pour n'afficher que les LLMs de génération de texte. La liste autorisée est définie dans `get_model_list()` :

```python
authorized_models = [
    "Apertus-70B-Instruct-2509",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "google/gemma-4-31B-it",
    "moonshotai/Kimi-K2.6",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    "mistralai/Mistral-Small-4-119B-2603",
]
```

Seuls les modèles présents à la fois dans cette liste **et** sur l'endpoint configuré sont proposés à la sélection.

---

## Paramètres disponibles

Accessibles dans la barre latérale pour les modes Complétion, Dialogue et Raisonnement :

| Paramètre | Défaut | Description |
|---|---|---|
| Modèle | premier disponible | Choix du LLM |
| Tokens à générer | 150 | Longueur max de la réponse |
| Température | 0.7 | Créativité (0 = déterministe, 1.5 = très créatif) |

---

## Fonctionnalités notables

- **Streaming token par token** en mode Complétion
- **Dialogue pré-amorcé éditable** — reproduit la technique de prompt engineering de 2022 sur les modèles de complétion
- **Prompt système visible et modifiable** en mode Raisonnement — les élèves voient exactement l'astuce
- **Compatible tout endpoint OpenAI-compatible** (DS platform, OpenAI, Mistral, Infomaniak, Ollama...)
- **SSL truststore** pour les environnements d'entreprise avec PKI interne

---

## Contexte

Développé pour un atelier de sensibilisation à l'IA générative pour lycéens, dans le cadre d'une démarche pédagogique visant à démystifier le fonctionnement des LLMs — de la complétion de texte jusqu'aux agents.