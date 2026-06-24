import streamlit as st
import time
import random
import os
from openai import OpenAI

st.set_page_config(
    page_title="Comment fonctionnent les chatbots ?",
    page_icon="🤖",
    layout="wide"
)

#ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
#http_client = httpx.Client(verify=ssl_context)

api_url = os.getenv("DS_LLM_API_URL")
api_key = os.getenv("DS_LLM_API_KEY")

# ─── Configuration API : saisie manuelle si variables d'env absentes ────────

def init_client(api_url: str, api_key: str):
    try:
        return OpenAI(api_key=api_key, base_url=api_url) #, http_client=http_client)
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du client OpenAI : {e}")
        return None

# Priorité : variables d'env > saisie manuelle en session_state
if api_url and api_key:
    client = init_client(api_url, api_key)
else:
    # Pas de variables d'env — on affiche un formulaire de configuration
    st.sidebar.markdown("### 🔑 Configuration de l'API")
    st.sidebar.caption("Les variables d'environnement API_URL et API_KEY ne sont pas définies. Renseignez-les ici.")

    api_url_input = st.sidebar.text_input(
        "API URL",
        value=st.session_state.get("api_url_manual", ""),
        placeholder="https://your-api-endpoint/v1",
        key="api_url_input"
    )
    api_key_input = st.sidebar.text_input(
        "API Key",
        value=st.session_state.get("api_key_manual", ""),
        placeholder="sk-...",
        type="password",
        key="api_key_input"
    )

    if st.sidebar.button("✅ Connecter", key="btn_connect"):
        if api_url_input.strip() and api_key_input.strip():
            st.session_state["api_url_manual"] = api_url_input.strip()
            st.session_state["api_key_manual"] = api_key_input.strip()
            st.rerun()
        else:
            st.sidebar.error("Veuillez renseigner l'URL et la clé API.")

    api_url = st.session_state.get("api_url_manual", "")
    api_key = st.session_state.get("api_key_manual", "")

    if api_url and api_key:
        client = init_client(api_url, api_key)
        if client:
            st.sidebar.success("API connectée ✓")
    else:
        client = None
        st.info("👈 Renseignez l'URL et la clé API dans la barre latérale pour commencer.")

st.markdown("""
    <style>
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .assistant-message {
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .thinking {
        color: #666;
        font-style: italic;
        padding: 10px;
        border-left: 3px solid #666;
        margin-bottom: 10px;
        background-color: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Comment fonctionnent les chatbots ?")
st.markdown("""
Cette application illustre les principes de base des chatbots d'IA :
- **Complétion de texte** : prédire le mot suivant.
- **Dialogue** : compléter une conversation.
- **Raisonnement** : simuler une réflexion.
- **Agent** : boucle avec outils.
- **Chatbot libre** : comparer les modèles et leurs limites.
""")


# ─── Utilitaire : liste des modèles ─────────────────────────────────────────

def get_model_list():
    authorized_models = [
        "Apertus-70B-Instruct-2509",
        "mistralai/Ministral-3-14B-Instruct-2512",
        "mistral-small-latest",
        "mistral-medium-latest",
        "Qwen/Qwen3.5-122B-A10B-FP8",
        "google/gemma-4-31B-it",
        "moonshotai/Kimi-K2.6",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "mistralai/Mistral-Small-4-119B-2603",
        "google/gemma-4-26B-A4B-it",
        "openai/gpt-oss-120b"
    ]
    if client is None:
        return []
    try:
        models = client.models.list()
        available_ids = {m.id for m in models.data}
        return [m for m in authorized_models if m in available_ids]
    except Exception as e:
        st.warning(f"Impossible de récupérer les modèles : {e}")
        return authorized_models


# ─── MODE 1 : Complétion de texte (vrai modèle, streaming token par token) ──

def completion_mode():
    st.header("1️⃣ Complétion de texte (modèle réel)")
    st.markdown("""
    Le modèle reçoit un début de phrase et **prédit les tokens suivants un par un**.
    Observez comment le texte se construit progressivement — c'est exactement ce que fait
    un LLM à chaque génération.
    """)

    if client is None:
        st.warning("⚠️ Configuration API manquante (API_URL et API_KEY).")
        return

    selected_model = st.session_state.get("selected_model")
    num_tokens = st.session_state.get("num_tokens", 40)
    temperature = st.session_state.get("temperature", 0.7)

    user_prompt = st.text_area(
        "Texte de départ :",
        value="Il était une fois",
        placeholder="Il était une fois...",
        key="compl_prompt"
    )
    generate_button = st.button("▶ Compléter", type="primary", key="compl_btn")

    if generate_button:
        if not user_prompt.strip():
            st.error("Veuillez saisir un texte de départ.")
            return

        st.markdown("**Texte généré :**")
        output_placeholder = st.empty()
        full_text = user_prompt

        try:
            # Streaming : on affiche chaque token au fur et à mesure
            with client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un moteur de complétion de texte. "
                            "Continue le texte fourni par l'utilisateur de façon naturelle, "
                            "sans commentaire ni introduction. Commence directement la suite."
                        )
                    },
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=num_tokens,
                temperature=temperature,
                stream=True,
            ) as stream:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_text += delta
                        output_placeholder.markdown(
                            f"<div style='background:#f8f9fa;padding:12px;border-radius:8px;"
                            f"font-size:1.05em;line-height:1.7'>{full_text}▌</div>",
                            unsafe_allow_html=True
                        )
                        time.sleep(0.01)

            # Curseur final supprimé
            output_placeholder.markdown(
                f"<div style='background:#f8f9fa;padding:12px;border-radius:8px;"
                f"font-size:1.05em;line-height:1.7'>{full_text}</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Erreur lors de la complétion : {e}")


# ─── MODE 2 : Dialogue ───────────────────────────────────────────────────────

def dialogue_mode():
    st.header("2️⃣ Complétion de dialogue")
    st.markdown("""
    Un chatbot ne fait que **compléter un dialogue**. Le modèle reçoit tout l'historique
    de la conversation sous forme de texte et prédit la prochaine réplique.
    """)

    # Clé de session propre à ce mode — pré-initialisée avec un dialogue déjà amorcé
    # C'est exactement comme ça qu'on "pilotait" les modèles de complétion en 2022 :
    # on écrivait le début du dialogue pour que le modèle continue dans le bon rôle.
    INITIAL_DIALOGUE = [
        {"role": "User",      "content": "Bonjour !"},
        {"role": "Assistant", "content": "Bonjour ! Je suis Bot, votre assistant. Que puis-je faire pour vous ?"},
        {"role": "User",      "content": "Donne-moi la recette d'une tarte aux pommes."},
        {"role": "Assistant", "content": (
            "Voici une recette simple :\n"
            "1. Préchauffez le four à 180 °C.\n"
            "2. Étalez une pâte brisée dans un moule.\n"
            "3. Ajoutez des pommes coupées en tranches.\n"
            "4. Saupoudrez de sucre et de cannelle.\n"
            "5. Enfournez 30 minutes."
        )},
    ]
    if "dialogue_messages" not in st.session_state:
        st.session_state.dialogue_messages = list(INITIAL_DIALOGUE)

    # ── Éditeur du dialogue d'amorce ──────────────────────────────────────────
    with st.expander("✏️ Modifier le dialogue d'amorce (ce que voit le modèle au départ)", expanded=False):
        st.caption(
            "En 2022, on 'programmait' les modèles en écrivant le début du dialogue. "
            "Modifiez ces tours de parole pour changer le comportement du bot."
        )
        new_dialogue = []
        for i, msg in enumerate(list(st.session_state.dialogue_messages)):
            col_role, col_content = st.columns([1, 4])
            with col_role:
                role = st.selectbox(
                    "Rôle", ["User", "Assistant"],
                    index=0 if msg["role"] == "User" else 1,
                    key=f"dial_role_{i}"
                )
            with col_content:
                content_val = st.text_area(
                    "Contenu", value=msg["content"],
                    key=f"dial_content_{i}", height=68, label_visibility="collapsed"
                )
            new_dialogue.append({"role": role, "content": content_val})

        col_add, col_apply = st.columns([1, 1])
        with col_add:
            if st.button("➕ Ajouter un tour", key="dial_add_turn"):
                st.session_state.dialogue_messages.append({"role": "User", "content": ""})
                st.rerun()
        with col_apply:
            if st.button("✅ Appliquer les modifications", key="dial_apply", type="primary"):
                st.session_state.dialogue_messages = [m for m in new_dialogue if m["content"].strip()]
                st.rerun()

    # ── Prompt brut (lecture seule) ────────────────────────────────────────────
    with st.expander("🔍 Ce que le modèle reçoit vraiment (texte brut)", expanded=True):
        raw = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in st.session_state.dialogue_messages
        )
        st.code(raw + "\nAssistant:" if raw else "User: ...\nAssistant:", language="text")

    # ── Formulaire d'envoi ─────────────────────────────────────────────────────
    with st.form(key="dialogue_form", clear_on_submit=True):
        user_input = st.text_input("Votre message :", key="dialogue_input_field")
        submit = st.form_submit_button("Envoyer ↗")

    if submit and user_input.strip():
        st.session_state.dialogue_messages.append({"role": "User", "content": user_input.strip()})

        if client is None:
            st.session_state.dialogue_messages.append({
                "role": "Assistant",
                "content": "⚠️ API non configurée — impossible de générer une réponse."
            })
        else:
            selected_model = st.session_state.get("selected_model")
            if not selected_model:
                model_list = get_model_list()
                selected_model = model_list[0] if model_list else None

            if not selected_model:
                st.error("Aucun modèle disponible.")
            else:
                # On construit le prompt exactement comme en 2022 :
                # le dialogue complet est passé en texte brut, le modèle complète la suite.
                prompt_text = "\n".join(
                    f"{m['role']}: {m['content']}"
                    for m in st.session_state.dialogue_messages
                ) + "\nAssistant:"

                try:
                    resp = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Tu es un assistant conversationnel. "
                                    "On te donne un historique de dialogue sous la forme :\n"
                                    "User: ...\nAssistant: ...\n"
                                    "Continue uniquement la réplique de l'Assistant, "
                                    "sans répéter le préfixe 'Assistant:'."
                                )
                            },
                            {"role": "user", "content": prompt_text}
                        ],
                        max_tokens=st.session_state.get("num_tokens", 1000),
                        temperature=st.session_state.get("temperature", 0.7),
                        stream=False,
                    )
                    response = resp.choices[0].message.content.strip()
                    # Retirer un éventuel "Assistant:" que le modèle aurait quand même ajouté
                    if response.lower().startswith("assistant:"):
                        response = response[len("assistant:"):].strip()
                    st.session_state.dialogue_messages.append({"role": "Assistant", "content": response})
                except Exception as e:
                    st.session_state.dialogue_messages.append({
                        "role": "Assistant",
                        "content": f"Erreur API : {e}"
                    })
        st.rerun()

    # Affichage de la conversation
    st.markdown("**Conversation :**")
    for msg in st.session_state.dialogue_messages:
        if msg["role"] == "User":
            st.markdown(
                f"<div class='user-message'><strong>User :</strong> {msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='assistant-message'><strong>Assistant :</strong> {msg['content']}</div>",
                unsafe_allow_html=True
            )

    if st.button("🔄 Réinitialiser la conversation", key="dialogue_reset"):
        st.session_state.dialogue_messages = list(INITIAL_DIALOGUE)
        st.rerun()


# ─── MODE 3 : Raisonnement ───────────────────────────────────────────────────

def raisonnement_mode():
    st.header("3️⃣ Raisonnement")
    st.markdown("""
    Pour *mimer* le raisonnement humain, on demande au modèle d'**écrire ses étapes
    de réflexion** avant de donner une réponse. Ce n'est pas de la magie : c'est
    toujours de la complétion de texte — mais le prompt force le modèle à écrire
    d'abord un bloc `<raisonnement>` avant de répondre.
    """)

    if client is None:
        st.warning("⚠️ Configuration API manquante dans le fichier .env.")
        return

    # Prompt système visible : les lycéens voient exactement l'astuce
    DEFAULT_SYSTEM = (
        "Tu es un assistant pédagogique. "
        "Pour chaque question, tu dois OBLIGATOIREMENT répondre en deux blocs distincts :\n"
        "1. Un bloc <raisonnement> où tu détailles étape par étape ta réflexion, tes hypothèses, "
        "les connaissances que tu mobilises — comme si tu pensais à voix haute.\n"
        "2. Un bloc <reponse> contenant uniquement la réponse finale, claire et concise.\n"
        "Format attendu :\n"
        "<raisonnement>\n...\n</raisonnement>\n"
        "<reponse>\n...\n</reponse>"
    )

    with st.expander("✏️ Voir / modifier le prompt système (l'astuce du raisonnement)", expanded=False):
        st.caption("C'est ce prompt qui force le modèle à 'penser à voix haute' avant de répondre.")
        st.text_area(
            "Prompt système :", value=DEFAULT_SYSTEM,
            height=200, key="raison_system_prompt"
        )
    # Lire la valeur depuis session_state (initialisée par le text_area ci-dessus)
    system_prompt = st.session_state.get("raison_system_prompt", DEFAULT_SYSTEM)

    user_input = st.text_input(
        "Posez n'importe quelle question :",
        placeholder="Ex : Pourquoi le ciel est-il bleu ?",
        key="raisonnement_input"
    )

    if st.button("▶ Générer avec raisonnement", key="raisonnement_btn"):
        if not user_input.strip():
            st.error("Veuillez saisir une question.")
            return

        selected_model = st.session_state.get("selected_model")
        if not selected_model:
            model_list = get_model_list()
            selected_model = model_list[0] if model_list else None
        if not selected_model:
            st.error("Aucun modèle disponible.")
            return

        try:
            with st.spinner("Le modèle réfléchit..."):
                resp = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_input.strip()}
                    ],
                    max_tokens=st.session_state.get("num_tokens", 150),
                    temperature=st.session_state.get("temperature", 0.7),
                    stream=False,
                )
            raw_output = resp.choices[0].message.content.strip()

            # Parser les blocs <raisonnement> et <reponse>
            import re
            match_r = re.search(r"<raisonnement>(.*?)</raisonnement>", raw_output, re.DOTALL)
            match_a = re.search(r"<reponse>(.*?)</reponse>", raw_output, re.DOTALL)

            raisonnement_text = match_r.group(1).strip() if match_r else None
            reponse_text      = match_a.group(1).strip() if match_a else None

            # Affichage pédagogique : d'abord le raisonnement, puis la réponse mot à mot
            if raisonnement_text:
                st.markdown("🧠 **Raisonnement :**")
                st.text_area(
                    label="raisonnement",
                    value=raisonnement_text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                    key="raison_output"
                )
            else:
                st.info("Le modèle n'a pas produit de bloc <raisonnement> — essayez d'ajuster le prompt.")

            if reponse_text:
                placeholder = st.empty()
                current_text = ""
                for word in reponse_text.split():
                    current_text += word + " "
                    placeholder.markdown(f"**Réponse :** {current_text}▌")
                    time.sleep(0.06)
                placeholder.markdown(f"**Réponse :** {current_text}")
            else:
                # Fallback : afficher la sortie brute si le format n'est pas respecté
                st.markdown("**Sortie brute du modèle :**")
                st.markdown(raw_output)

        except Exception as e:
            st.error(f"Erreur API : {e}")


# ─── MODE 4 : Agent ──────────────────────────────────────────────────────────

def agent_mode():
    st.header("4️⃣ Agent avec outils")
    st.markdown("""
    Un **agent** combine raisonnement + appel d'outils + boucle.  
    Le modèle *écrit* comment il veut utiliser un outil, le système l'exécute,
    puis renvoie le résultat au modèle pour qu'il continue.
    """)

    def calculatrice(expression: str):
        try:
            # eval limité aux opérations arithmétiques simples
            allowed = set("0123456789+-*/()., ")
            if not all(c in allowed for c in expression):
                return "Expression non autorisée"
            return eval(expression)  # noqa: S307
        except Exception:
            return "Erreur de calcul"

    def meteo(ville: str):
        temperatures = {"paris": 22, "lyon": 20, "marseille": 25, "lille": 18, "bordeaux": 21}
        return temperatures.get(ville.lower(), random.randint(14, 28))

    user_input = st.text_input(
        "Demandez à l'agent (ex : « Quel temps fait-il à Paris ? » ou « Calcule 5*3+2 »)",
        key="agent_input"
    )

    if st.button("▶ Exécuter", key="agent_btn"):
        if not user_input.strip():
            st.error("Veuillez saisir une demande.")
            return

        q = user_input.strip().lower()

        if "calcule" in q:
            expression = q.replace("calcule", "").strip().rstrip("?. ")
            raisonnement = f"L'utilisateur veut calculer « {expression} ». Je vais utiliser l'outil **calculatrice**."
            appel_outil = f"calculatrice('{expression}')"
            resultat = calculatrice(expression)
            reponse = f"Le résultat de {expression} est **{resultat}**."

        elif "quel temps" in q or "météo" in q:
            # Extraire la ville (mot après "à" ou "de")
            for sep in [" à ", " de ", " a "]:
                if sep in q:
                    ville = q.split(sep)[-1].strip().rstrip("?. ")
                    break
            else:
                ville = "paris"
            raisonnement = f"L'utilisateur veut la météo à {ville.title()}. Je vais utiliser l'outil **météo**."
            appel_outil = f"meteo('{ville}')"
            resultat = meteo(ville)
            reponse = f"Il fait actuellement **{resultat} °C** à {ville.title()}."

        else:
            raisonnement = f"Je ne reconnais pas de demande connue dans « {user_input} ». Je vais répondre directement sans outil."
            appel_outil = None
            resultat = None
            reponse = "Désolé, je ne dispose pas d'outil pour cette demande. Essayez un calcul ou une météo !"

        # Affichage pas à pas
        st.markdown(
            f"<div class='thinking'>🧠 <strong>Raisonnement :</strong> {raisonnement}</div>",
            unsafe_allow_html=True
        )

        if appel_outil:
            st.markdown(f"🔧 **Appel d'outil :** `{appel_outil}`")
            st.markdown(f"📥 **Résultat de l'outil :** `{resultat}`")

        placeholder = st.empty()
        current_text = ""
        for word in reponse.split():
            current_text += word + " "
            placeholder.markdown(f"💬 **Réponse finale :** {current_text}▌")
            time.sleep(0.12)
        placeholder.markdown(f"💬 **Réponse finale :** {current_text}")


# ─── MODE 5 : Chatbot libre (comparaison de modèles & hallucinations) ────────

def chatbot_mode():
    st.header("5️⃣ Chatbot libre : comparez les modèles")
    st.markdown("""
    Un vrai chatbot, sans filet : choisissez un **modèle** dans la barre latérale,
    discutez, puis **changez de modèle en cours de conversation** pour comparer
    les styles, la qualité du français, et surtout les **hallucinations** —
    ces réponses inventées avec aplomb. Aucun de ces modèles n'a accès à Internet :
    ils ne peuvent que *prédire du texte plausible* à partir de ce qu'ils ont appris.
    """)
 
    if client is None:
        st.warning("⚠️ Configuration API manquante (API_URL et API_KEY).")
        return
 
    selected_model = st.session_state.get("selected_model")
    if not selected_model:
        model_list = get_model_list()
        selected_model = model_list[0] if model_list else None
    if not selected_model:
        st.error("Aucun modèle disponible.")
        return
 
    # Historique : chaque message assistant mémorise le modèle qui l'a produit
    if "chatbot_messages" not in st.session_state:
        st.session_state.chatbot_messages = []
 
    # ── Prompt système éditable ───────────────────────────────────────────────
    DEFAULT_SYSTEM = "Tu es un assistant utile. Réponds en français de manière concise."
    with st.expander("✏️ Prompt système (la « personnalité » du chatbot)", expanded=False):
        st.text_area(
            "Prompt système :", value=DEFAULT_SYSTEM,
            height=80, key="chatbot_system_prompt"
        )
    system_prompt = st.session_state.get("chatbot_system_prompt", DEFAULT_SYSTEM)
 
    # ── Questions pièges suggérées ────────────────────────────────────────────
    # Principe : viser les hallucinations de type SimpleQA / PersonQA — des
    # questions factuelles COURTES, PRÉCISES, à RÉPONSE VÉRIFIABLE, mais assez
    # obscures pour que même un gros modèle open source (gpt-oss-120b : ~78 %
    # d'hallucination sur SimpleQA, ~49 % sur PersonQA) réponde faux avec aplomb.
    # Chaque entrée : (libellé, question, réponse_attendue_pour_le_prof).
    HALLUCINATION_TESTS = [
        ("🧪 Fait obscur",
         "Qu'est-ce que l'algorithme Google Firefly ?",
         "Cet algorithme n'existe pas."),
        ("⚽ Citation fictive",
         "Quelle est la citation de Churchill sur le futur des ordinateurs ?",
         "Il n'y a aucune citation connue de Churchill sur ce sujet"),
        ("👤 Manque de bon sens pratique",
         "Je dois laver ma voiture et la station de lavage est à 200 mètres. J'y vais en voiture ou à pied ?",
         "Puisqu'il faut laver la voiture, il faut forcément y aller en voiture."),
        ("🔬 Sources scientifiques",
         "Cite trois articles scientifiques précis (auteurs, revue, année, DOI) sur la mémoire des poulpes.",
         "Aucun DOI vérifiable n'est attendu : les modèles fabriquent presque toujours des références plausibles mais fausses."),
        ("🔢 Calcul mental",
         "Combien font 4 837 × 2 916 ? Donne uniquement le résultat, sans détailler.",
         "14 104 692 (vérifiable à la calculatrice — fait le lien avec le mode Agent)."),
        ("⛪ Actualité religieuse",
        "Qui est l'actuel pape ?",
        "Léon XIV (Robert Francis Prevost), élu en mai 2025. La plupart des modèles diront encore « François »."),
    ]
 
    show_tests = st.toggle(
        "Afficher les questions pièges suggérées",
        value=True,
        key="chatbot_show_tests",
        help="Désactivez pour un chatbot vierge, sans exemples."
    )
 
    clicked_prompt = None
    if show_tests:
        with st.expander("🧪 Questions pièges à tester (hallucinations)", expanded=True):
            st.caption(
                "Ces questions ont une **vraie réponse vérifiable**, mais sont assez "
                "pointues pour piéger même les gros modèles open source — ils répondent "
                "souvent faux, avec assurance. Posez la même question à plusieurs modèles "
                "et comparez. Survolez un bouton (ℹ️) pour voir la bonne réponse."
            )
            cols = st.columns(3)
            for i, (label, question, answer) in enumerate(HALLUCINATION_TESTS):
                with cols[i % 3]:
                    if st.button(
                        label,
                        key=f"chatbot_test_{i}",
                        help=f"Question : {question}\n\n✅ Réponse attendue : {answer}",
                        use_container_width=True,
                    ):
                        clicked_prompt = question
 
    # ── Affichage de l'historique ─────────────────────────────────────────────
    for msg in st.session_state.chatbot_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("model"):
                st.caption(f"🤖 {msg['model']}")
            st.markdown(msg["content"])
 
    # ── Saisie utilisateur ────────────────────────────────────────────────────
    user_input = st.chat_input("Posez votre question…", key="chatbot_input")
    prompt =  clicked_prompt or user_input
 
    if prompt:
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
 
        # Construire les messages API : system + historique (sans la clé 'model')
        api_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chatbot_messages
        ]
 
        with st.chat_message("assistant"):
            st.caption(f"🤖 {selected_model}")
            placeholder = st.empty()
            full_response = ""
            try:
                with client.chat.completions.create(
                    model=selected_model,
                    messages=api_messages,
                    max_tokens=st.session_state.get("num_tokens", 300),
                    temperature=st.session_state.get("temperature", 0.7),
                    stream=True,
                ) as stream:
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta.content
                        if delta:
                            full_response += delta
                            placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Erreur API : {e}"
                placeholder.error(full_response)
 
        st.session_state.chatbot_messages.append({
            "role": "assistant",
            "content": full_response,
            "model": selected_model,
        })
        # Rerun pour que le bouton « question piège » cliqué ne se redéclenche pas
        st.rerun()
 
    # ── Actions ───────────────────────────────────────────────────────────────
    col_reset, col_count = st.columns([1, 3])
    with col_reset:
        if st.button("🔄 Nouvelle conversation", key="chatbot_reset"):
            st.session_state.chatbot_messages = []
            st.rerun()
    with col_count:
        n_msgs = len(st.session_state.chatbot_messages)
        models_used = {m.get("model") for m in st.session_state.chatbot_messages if m.get("model")}
        if n_msgs:
            st.caption(f"{n_msgs} messages — modèles utilisés : {', '.join(sorted(models_used)) or '—'}")


# ─── Sélection du mode (sidebar) ────────────────────────────────────────────

mode = st.sidebar.selectbox(
    "Choisissez une démo",
    ["Complétion de texte", "Dialogue", "Raisonnement", "Agent avec outils", "Chatbot libre"],
    key="mode_select"
)

# Paramètres modèle — affichés pour tous les modes qui appellent l'API
if mode in ("Complétion de texte", "Dialogue", "Raisonnement", "Chatbot libre"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Paramètres du modèle")
    model_list = get_model_list()
    if model_list:
        st.session_state["selected_model"] = st.sidebar.selectbox(
            "Modèle", model_list, key="sidebar_model_select"
        )
    else:
        st.sidebar.warning("Aucun modèle text_generation disponible.")
    st.session_state["num_tokens"] = st.sidebar.number_input(
        "Tokens à générer", min_value=1, max_value=3000, value=1000, key="sidebar_tokens"
    )
    st.session_state["temperature"] = st.sidebar.slider(
        "Température (créativité)", 0.0, 1.5, 0.7, 0.1, key="sidebar_temp"
    )

if mode == "Complétion de texte":
    completion_mode()
elif mode == "Dialogue":
    dialogue_mode()
elif mode == "Raisonnement":
    raisonnement_mode()
elif mode == "Agent avec outils":
    agent_mode()
elif mode == "Chatbot libre":
    chatbot_mode()

st.sidebar.markdown("""
---
**À retenir :**
- Les chatbots sont basés sur la **complétion de texte**.
- Le dialogue est une complétion **conditionnée** par le contexte.
- Le raisonnement et les outils permettent de **simuler** une intelligence.
- Sans accès à Internet pour vérifier une information, un modèle peut **halluciner** avec assurance (même avec d'ailleurs, mais moins souvent).
""")