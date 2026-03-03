"""
tasks/domain.py — Domain shift evaluation (perplexity + QA).

Standalone — no imports from other project files.
"""

from __future__ import annotations

import math
import random

import torch
from tqdm import tqdm

_SCIENCE_QA = [
    # Biology
    ("What organelle is responsible for ATP production in eukaryotic cells?", "mitochondria"),
    ("What molecule carries genetic information in most organisms?", "dna"),
    ("What is the chemical formula for water?", "h2o"),
    ("What pigment gives plants their green color?", "chlorophyll"),
    ("What type of cell division produces gametes?", "meiosis"),
    ("What is the basic structural and functional unit of life?", "cell"),
    ("What polymer is made of amino acid monomers?", "protein"),
    # Physics
    ("What force keeps planets in orbit around the sun?", "gravity"),
    ("What particle carries a positive charge in an atom?", "proton"),
    ("What is the speed of light in a vacuum in m/s?", "299792458"),
    ("What is the SI unit of electric current?", "ampere"),
    ("What law states that energy cannot be created or destroyed?", "thermodynamics"),
    ("What subatomic particle has no electric charge?", "neutron"),
    ("What quantity is measured in hertz?", "frequency"),
    # Chemistry
    ("What bond involves the sharing of electron pairs between atoms?", "covalent"),
    ("What is the pH of a neutral solution at 25°C?", "7"),
    ("What is the most abundant element in the universe?", "hydrogen"),
    ("What is the atomic number of carbon?", "6"),
    ("What type of reaction releases energy to the surroundings?", "exothermic"),
    ("How many electrons can the first electron shell hold?", "2"),
    # Earth science & astronomy
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen"),
    ("What scale measures the magnitude of earthquakes?", "richter"),
    ("What layer of the atmosphere contains the ozone layer?", "stratosphere"),
    ("What is the closest star to Earth?", "sun"),
    ("What type of rock is formed from cooled magma?", "igneous"),
    # Medicine & neuroscience
    ("What organ in the human body produces insulin?", "pancreas"),
    ("What protein carries oxygen in red blood cells?", "hemoglobin"),
    ("What neurotransmitter is primarily associated with reward?", "dopamine"),
    ("What is the largest organ of the human body?", "skin"),
    ("What type of blood cell fights infections?", "leukocyte"),
]

_FALLBACK_TEXTS = [
    # Biology
    "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells."
    " They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source"
    " of chemical energy.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into"
    " chemical energy stored in carbohydrate molecules. The light-dependent reactions occur in the"
    " thylakoid membranes, while the Calvin cycle takes place in the stroma of chloroplasts.",
    "Protein folding is the physical process by which a protein chain acquires its native 3D"
    " structure. Misfolded proteins can aggregate and are associated with diseases such as"
    " Alzheimer's and Parkinson's.",
    "CRISPR-Cas9 is a genome editing tool that allows researchers to alter DNA sequences and modify"
    " gene function with high precision. The system relies on a guide RNA to direct the Cas9"
    " nuclease to a specific genomic locus.",
    "The human gut microbiome comprises trillions of microorganisms that play essential roles in"
    " digestion, immune function, and metabolite production. Dysbiosis of gut flora has been linked"
    " to inflammatory bowel disease and metabolic disorders.",
    # Physics
    "General relativity describes gravity as the curvature of spacetime caused by mass and energy."
    " Massive objects like stars and black holes warp the fabric of spacetime, causing nearby"
    " objects to follow curved trajectories.",
    "Quantum entanglement is a phenomenon in which two particles become correlated such that the"
    " quantum state of one instantly influences the state of the other, regardless of the distance"
    " separating them.",
    "Superconductivity is a state of zero electrical resistance observed in certain materials below"
    " a critical temperature. In this state, electric current flows without energy dissipation and"
    " magnetic flux is expelled from the material via the Meissner effect.",
    # Chemistry
    "Catalysts accelerate chemical reactions by lowering the activation energy without being"
    " consumed in the process. Enzymes are biological catalysts that exhibit remarkable"
    " specificity, often catalyzing a single reaction type.",
    "Electronegativity is the tendency of an atom to attract shared electrons toward itself in a"
    " chemical bond. Fluorine has the highest electronegativity of all elements, which explains"
    " its strong oxidizing behavior.",
    "The Haber-Bosch process synthesizes ammonia from atmospheric nitrogen and hydrogen under high"
    " pressure and temperature using an iron catalyst. This industrial process is responsible for"
    " producing the majority of the world's fertilizer.",
    # Medicine
    "The blood-brain barrier is a highly selective semipermeable membrane that separates"
    " circulating blood from the brain extracellular fluid. Tight junctions between endothelial"
    " cells restrict"
    " the passive diffusion of most molecules into the central nervous system.",
    "Messenger RNA vaccines work by delivering synthetic mRNA encoding a viral antigen into host"
    " cells. The ribosomes translate the mRNA into protein, which triggers an adaptive immune"
    " response without using live virus.",
    "Antibiotic resistance arises when bacteria evolve mechanisms to survive exposure to"
    " antimicrobial drugs. Horizontal gene transfer through plasmids allows resistance genes to"
    " spread rapidly"
    " across bacterial populations.",
    # Computer science
    "Gradient descent is an iterative optimization algorithm used to minimize a differentiable"
    " loss function. The parameters are updated in the direction of the negative gradient,"
    " scaled by a learning rate that controls step size.",
    "Transformer architectures rely on self-attention mechanisms that compute pairwise interactions"
    " between all tokens in a sequence. This allows the model to capture long-range dependencies"
    " without recurrence or convolution.",
    "Hash tables provide average-case constant-time lookup by mapping keys to array indices via"
    " a hash function. Collision resolution strategies such as chaining and open addressing"
    " determine performance under high load factors.",
    # Earth science
    "Plate tectonics describes the movement of lithospheric plates on the asthenosphere driven by"
    " mantle convection. Divergent boundaries create new crust at mid-ocean ridges, while"
    " convergent boundaries produce subduction zones and mountain ranges.",
    "The carbon cycle describes the movement of carbon between the atmosphere, oceans, biosphere,"
    " and lithosphere. Anthropogenic emissions from fossil fuel combustion have increased"
    " atmospheric CO2 concentrations well above pre-industrial levels.",
    # Astronomy
    "Neutron stars are the collapsed cores of massive stars that have undergone supernova"
    " explosions. With masses up to twice that of the Sun compressed into a sphere roughly"
    " 10 kilometers across,"
    " they exhibit extreme gravitational and magnetic fields.",
    "The cosmic microwave background radiation is the thermal remnant of the early universe,"
    " emitted roughly 380,000 years after the Big Bang when protons and electrons first combined"
    " to form neutral hydrogen atoms.",
    # Neuroscience
    "Long-term potentiation is a persistent strengthening of synaptic connections following"
    " high-frequency stimulation. It is widely considered one of the major cellular mechanisms"
    " underlying learning and memory formation in the hippocampus.",
    "The prefrontal cortex is involved in executive functions including decision-making, working"
    " memory, and impulse control. Lesions to this region are associated with deficits in planning"
    " and social behavior.",
]


@torch.no_grad()
def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate domain adaptation.

    Returns: {
        "domain_ppl": float,         # held-out perplexity
        "domain_qa_accuracy": float,  # domain QA exact-match
    }
    """
    # ── Perplexity on scientific text ────────────────────────────────────
    n_ppl = 3 if mode == "smoke" else 100 if mode == "fast" else 500
    texts = _load_perplexity_texts(n_ppl)

    total_loss = 0.0
    total_tokens = 0
    for text in tqdm(texts, desc="domain-ppl"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        out = model(**enc, labels=labels)
        n_tok = labels.numel()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok

    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    # ── Domain QA (SciQ dataset, fallback to hardcoded) ─────────────────
    n_qa = 5 if mode == "smoke" else 200 if mode == "fast" else None
    qa_items: list[tuple[str, str]] = []
    try:
        from datasets import load_dataset

        sciq = load_dataset("allenai/sciq", split="test")
        for ex in sciq:
            q = ex.get("question", "")
            a = ex.get("correct_answer", "")
            if q and a:
                qa_items.append((q, a))
        if n_qa is not None:
            qa_items = qa_items[:n_qa]
    except Exception:
        qa_items = _SCIENCE_QA[:3] if mode == "smoke" else _SCIENCE_QA

    qa_per_example: list[float] = []
    for question, answer in tqdm(qa_items, desc="domain-qa"):
        prompt = f"Answer in one or two words.\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

        qa_per_example.append(1.0 if answer.lower() in response else 0.0)

    qa_acc = sum(qa_per_example) / len(qa_per_example) if qa_per_example else 0.0

    return {
        "domain_ppl": ppl,
        "domain_qa_accuracy": qa_acc,
        "domain_qa_accuracy_scores": qa_per_example,
    }


def _load_perplexity_texts(n: int) -> list[str]:
    """Load perplexity evaluation texts with tiered fallback.

    Priority: allenai/sciq support field → scientific_papers → hardcoded.
    """
    # Try 1: allenai/sciq (small, reliable, already cached for QA)
    try:
        from datasets import load_dataset

        sciq = load_dataset("allenai/sciq", split="test")
        texts = []
        for ex in sciq:
            support = ex.get("support", "")
            if support and len(support) > 50:
                texts.append(support[:2048])
            if len(texts) >= n:
                break
        if texts:
            return texts[:n]
    except Exception:
        pass

    # Try 2: scientific_papers (large, often fails to download)
    try:
        from datasets import load_dataset

        ds = load_dataset("scientific_papers", "pubmed", split="test", streaming=True)
        texts = []
        for i, ex in enumerate(ds):
            if i >= n:
                break
            abstract = ex.get("abstract", ex.get("text", ""))
            if abstract and len(abstract) > 50:
                texts.append(abstract[:2048])
        if texts:
            return texts[:n]
    except Exception:
        pass

    # Try 3: hardcoded fallback with shuffled deduplication
    # Shuffle copies with different seeds so repeated texts appear in
    # varied tokenization contexts (preceding text differs each cycle).
    n_base = len(_FALLBACK_TEXTS)
    if n <= n_base:
        return _FALLBACK_TEXTS[:n]

    texts = list(_FALLBACK_TEXTS)
    copies_needed = (n // n_base + 1) - 1  # already have 1 copy
    for i in range(copies_needed):
        chunk = list(_FALLBACK_TEXTS)
        random.Random(42 + i).shuffle(chunk)
        texts.extend(chunk)
    return texts[:n]
