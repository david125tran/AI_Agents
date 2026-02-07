# ./wardrobe_handbook_kb.py

# ---------------------------------- Imports ----------------------------------
from colorama import Fore, Style, init
init(autoreset=True)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------ Functions ----------------------------------
def print_banner(text: str, *, color: str = Fore.LIGHTWHITE_EX, width: int = 65) -> None:
    """
    Print a banner with the given text for visual separation in console output.
    color should be a colorama Fore.* value (e.g., Fore.CYAN).
    """
    line = f"{color}* {Style.RESET_ALL}" * width
    print("\n")
    print(line)
    print(f"{color}*{Style.RESET_ALL} {color}{text}{Style.RESET_ALL}")
    print(line)
    print("\n")


def classify_article_bin(df, article_col="Article"):
    """
    Classifies each clothing article as 'Tops', 'Bottoms', or 'Footwear' using TF-IDF
    nearest-reference anchors. 

    This function takes the items in the referenced article_col and assigns a Bin.  
    It then bins items from the wardrobe by doing a nearest-neighbor match against
    predefined reference articles for each bin.
    """
    # Different clothing articles to serve as references for each bin
    reference = {
        "Tops": [
            "shirt", "t shirt", "tee", "polo", "button down",
            "sweater", "cardigan", "hoodie", "blazer", "jacket",
            "coat", "overcoat", "parka", "vest", "waistcoat"
        ],
        "Bottoms": [
            "pant", "pants", "jean", "jeans",
            "chino", "chinos", "trouser", "trousers",
            "slack", "slacks", "short", "shorts",
            "legging", "leggings", "trunk", "trunks"
        ],
        "Footwear": [
            "shoe", "shoes", "sneaker", "sneakers",
            "boot", "boots", "loafers", "loafer", 
            "sandal", "sandals", "flip flop", "slipper", "slippers",
            "crocs",  "cap-toe"
        ]
    }

    # Flatten reference articles and keep mapping
    ref_articles = []
    ref_bins = []
    for bin_name, articles in reference.items():
        for a in articles:
            ref_articles.append(a.lower())
            ref_bins.append(bin_name)

    # Prepare vectorizer (fit on refs + actual article texts)
    vectorizer = TfidfVectorizer().fit(ref_articles + df[article_col].astype(str).str.lower().tolist())

    # Embed reference articles
    ref_vecs = vectorizer.transform(ref_articles)

    # Classify each article via TF-IDF nearest reference
    bins = []
    print_banner("Text Classification of Clothing Articles into Bins: Tops, Bottoms, Footwear using TF-IDF Nearest Reference")

    for article in df[article_col].astype(str):
        a_low = article.lower()
        vec = vectorizer.transform([a_low])
        sims = cosine_similarity(vec, ref_vecs)[0]
        best_idx = sims.argmax()
        bin_name = ref_bins[best_idx]
        if bin_name == "Tops":
            # Make 'Tops' cyan
            print(f"Placing '{Fore.LIGHTYELLOW_EX}{article}{Style.RESET_ALL}' into the '{Fore.CYAN}Tops 👕{Style.RESET_ALL}' cabinet.")
        elif bin_name == "Bottoms":
            # Make 'Bottoms' green
            print(f"Placing '{Fore.LIGHTYELLOW_EX}{article}{Style.RESET_ALL}' into the '{Fore.GREEN}Bottoms 👖{Style.RESET_ALL}' cabinet.")
        elif bin_name == "Footwear":
            # Make 'Footwear' yellow
            print(f"Placing '{Fore.LIGHTYELLOW_EX}{article}{Style.RESET_ALL}' into the '{Fore.RED}Footwear 👟{Style.RESET_ALL}' cabinet.")

        bins.append(bin_name)

    print(f"\n{Fore.LIGHTYELLOW_EX}All clothing sorted into cabinets for faster retrieval!\n{Style.RESET_ALL}")
    df["Bin"] = bins
    return df


def evaluate_binning_confusion(df_raw: pd.DataFrame, df_pred: pd.DataFrame) -> None:
    """
    Compare ground-truth Cabinet vs predicted Bin using a confusion matrix.
    df_raw must contain Cabinet; df_pred must contain Bin in same row order.
    """
    if "Cabinet" not in df_raw.columns:
        print("No 'Cabinet' column found — can't compute confusion matrix.")
        return
    if "Bin" not in df_pred.columns:
        print("No 'Bin' column found in predictions — can't compute confusion matrix.")
        return

    y_true = df_raw["Cabinet"].astype(str).str.strip().replace({
        "Footware": "Footwear",
        "footware": "Footwear",
        "footwear": "Footwear",
        "tops": "Tops",
        "bottoms": "Bottoms",
    })

    y_pred = df_pred["Bin"].astype(str).str.strip()

    print_misclassified_rows(df_raw, df_pred, max_rows=200)

    labels = ["Tops", "Bottoms", "Footwear"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print_banner("📊 Confusion Matrix: Cabinet (truth) vs Bin (pred)", color=Fore.MAGENTA)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print(cm_df.to_string())

    print("\n" + Fore.CYAN + "Classification report:" + Style.RESET_ALL)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


def print_misclassified_rows(df_raw: pd.DataFrame, df_pred: pd.DataFrame, *, max_rows: int = 50):
    """
    This function identifies and prints rows where the predicted 'Bin' does not match the true 'Cabinet'.
     It shows the Article, Cabinet, Predicted Bin, and other relevant metadata for each misclassified item.
    """
    merged = df_raw.copy()

    # Ensure we have truth + prediction
    if "Cabinet" not in merged.columns:
        print("No 'Cabinet' column found — can't list misclassified rows.")
        return
    if "Bin" not in df_pred.columns:
        print("No 'Bin' column found in predictions — can't list misclassified rows.")
        return

    merged["Predicted"] = df_pred["Bin"].astype(str).str.strip()

    # Normalize truth labels
    merged["Cabinet"] = (
        merged["Cabinet"]
        .astype(str)
        .str.strip()
        .replace({
            "Footware": "Footwear",
            "footware": "Footwear",
            "footwear": "Footwear",
            "tops": "Tops",
            "bottoms": "Bottoms",
        })
    )

    wrong = merged[merged["Cabinet"] != merged["Predicted"]].copy()

    print_banner(
        f"❌ Misclassified items: {len(wrong)} (showing {min(len(wrong), max_rows)})",
        color=Fore.RED
    )

    if wrong.empty:
        print("None 🎉")
        return

    # Show useful columns if they exist
    cols = ["Article", "Cabinet", "Predicted", "Style", "Color", "Pattern", "Tags/Metadata", "Work Appropriate"]
    cols = [c for c in cols if c in wrong.columns]

    wrong = wrong.sort_values(["Cabinet", "Predicted", "Article"])
    print(wrong[cols].head(max_rows).to_string(index=False))


def wardrobe_json_to_toon(results: list[dict]) -> str:
    """
    Convert wardrobe search results (list of dicts) into a compact
    Toon-style CSV string for token-efficient LLM prompts.
    """

    if not results:
        return "article,style,color,pattern,bin,work_appropriate\n"

    header = "article,style,color,pattern,bin,work_appropriate"
    lines = [header]

    for item in results:
        line = ",".join([
            str(item.get("Article", "")),
            str(item.get("Style", "")),
            str(item.get("Color", "")),
            str(item.get("Pattern", "")),
            str(item.get("Bin", "")),
            str(item.get("Work Appropriate", ""))
        ])
        lines.append(line)

    return "\n".join(lines)


def read_wardrobe_knowledge_base(WARDROBE_KNOWLEDGE_BASE):
    """
    Reads the wardrobe knowledge base from the Excel file.
    Returns a list of dicts representing the wardrobe items.
    """

    df = pd.read_excel(WARDROBE_KNOWLEDGE_BASE)
    return df.to_dict(orient="records")