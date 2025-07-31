# from `gold_with_3_distractors_context_cot_qa_codex.txt`

one_shot_rag_qa_docs = (
    "A novel keratin 5 mutation (K5V186L) in a family with EBS-K: a conservative substitution can lead to development of different disease phenotypes."
    "\nEpidermolysis bullosa simplex is a hereditary skin blistering disorder caused by mutations in the KRT5 or KRT14 genes."
    "\n\nKeratin mutations in patients with epidermolysis bullosa simplex: correlations between phenotype severity and disturbance of intermediate filament molecular structure."
    "\nEpidermolysis bullosa simplex (EBS) is an inherited skin disorder caused by mutations in the keratin 5 (KRT5) and keratin 14 (KRT14) genes, with fragility of basal keratinocytes leading to epidermal cytolysis and blistering."
    "\nIn this study, we characterized mutations in KRT5 and KRT14 genes in patients with EBS and investigated their possible structure-function correlations."
    "\n\nA new pathogenic keratin 5 mutation in a Hindoestan family with localized epidermolysis bullosa simplex."
    "\nEpidermolysis bullosa simplex is an autosomal dominant inherited skin blistering disorder caused by mutations in the genes KRT5 or KRT14 coding for the basal epidermal keratins 5 and 14, respectively."
    "\n\nMutation analysis of the entire keratin 5 and 14 genes in patients with epidermolysis bullosa simplex and identification of novel mutations."
    "\nEpidermolysis bullosa simplex is a group of blistering skin disorders caused by defects in one of the keratin genes, KRT5 and KRT14."
    "\n\nIdentification of novel and known KRT5 and KRT14 mutations in 53 patients with epidermolysis bullosa simplex: correlation between genotype and phenotype."
    "\nBACKGROUND Basal epidermolysis bullosa simplex (EBS) is a hereditary skin blistering disorder resulting in most cases from missense mutations in the keratin 5 (KRT5) or keratin 14 (KRT14) genes."
)


rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive \"yes\" or \"no\" answer to the question based on your analysis. '
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "Are keratin 8 Y54H and G62C mutations associated with inflammatory bowel disease?"
    '\nThought: '
)

one_shot_rag_qa_output = (
    "Our data indicate that both keratin 8 mutations, G62C and Y54H, do not play a relevant pathogenic role in inflammatory bowel disease. "
    "\nAnswer: no"
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]
