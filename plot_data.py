import json
import spacy
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
from typing import List, Dict, Tuple
import plotly.express as px


def load_dataset(file_path: str) -> List[Dict]:
    """Load the Alpaca-format dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_verb_and_object(instruction: str, nlp) -> Tuple[str, str]:
    """Extract root verb and its direct object using spaCy"""
    doc = nlp(instruction)
    root_verb = None
    direct_object = None
    
    # Find root verb
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root_verb = token.lemma_.lower()
            # Find direct object
            for child in token.children:
                if child.dep_ == "dobj":
                    # Get the full noun phrase if possible
                    direct_object = ' '.join([tok.text for tok in child.subtree]).lower()
                    break
            break
    
    return root_verb or "other", direct_object or "other"

def analyze_instructions(data: List[Dict], nlp) -> Dict:
    """Analyze instructions to get verb-object hierarchy"""
    hierarchy = defaultdict(lambda: defaultdict(int))
    
    for item in data:
        verb, obj = extract_verb_and_object(item['instruction'], nlp)
        hierarchy[verb][obj] += 1
    
    return hierarchy

def create_sunburst_chart(hierarchy: Dict, min_count: int = 5):
    """Create a sunburst chart similar to Self-Instruct paper Figure 2"""
    labels = []
    parents = []
    values = []
    
    # Filter and sort verbs by frequency
    verb_totals = {verb: sum(objs.values()) for verb, objs in hierarchy.items()}
    sorted_verbs = sorted(verb_totals.items(), key=lambda x: x[1], reverse=True)
    
    # Add root
    labels.append("instructions")
    parents.append("")
    values.append(sum(verb_totals.values()))
    
    # Add verbs and their objects
    for verb, total in sorted_verbs:
        if total >= min_count:  # Filter out rare verbs
            labels.append(verb)
            parents.append("instructions")
            values.append(total)
            
            # Sort objects by frequency
            sorted_objects = sorted(hierarchy[verb].items(), key=lambda x: x[1], reverse=True)
            for obj, count in sorted_objects:
                if count >= min_count:  # Filter out rare objects
                    labels.append(f"{verb}_{obj}")
                    parents.append(verb)
                    values.append(count)
    
    # Create the sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        maxdepth=2,
        insidetextorientation='radial',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
    ))
    
    # Update layout to match paper style
    fig.update_layout(
        width=1000,
        height=1000,
        title={
            'text': "Instruction Diversity Analysis",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        sunburstcolorway=px.colors.qualitative.Set3,  # Use a softer color palette
        margin=dict(t=100, l=0, r=0, b=0)
    )
    
    return fig

def main(file_path: str):
    """Main function to process dataset and create visualization"""
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Load and process data
    data = load_dataset(file_path)
    hierarchy = analyze_instructions(data, nlp)
    
    # Create and save visualization
    fig = create_sunburst_chart(hierarchy)
    fig.write_html("instruction_diversity.html")
    
    # Print some statistics
    print(f"Total instructions analyzed: {len(data)}")
    print("\nMost common instruction patterns:")
    for verb, objects in sorted(hierarchy.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
        total = sum(objects.values())
        print(f"\n{verb.upper()} ({total} total):")
        for obj, count in sorted(objects.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {obj}: {count}")

if __name__ == "__main__":
    # Install required packages if needed
    import sys
    import subprocess
    required_packages = ['spacy', 'plotly', 'pandas']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    file_path = "data/alpaca_data.json"  # Replace with your dataset path
    main(file_path)