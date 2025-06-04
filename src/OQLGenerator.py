from langchain_openai import AzureChatOpenAI
import os

context_prompt = """
You are an expert in cBioPortal's Onco Query Language (OQL). Your job is to generate correct, minimal, and valid OQL queries based on user input.

Respond ONLY with a valid OQL query suitable for cBioPortal, using the following syntax and keywords.

# Syntax Format:
GENE: OQL_KEYWORDS;

# OQL Keywords:
- MUT: all non-synonymous mutations
  - MUT = <protein change> (e.g., V600E)
  - MUT = <mutation type> (MISSENSE, NONSENSE, NONSTART, NONSTOP, FRAMESHIFT, INFRAME, SPLICE, TRUNC)
- FUSION: all gene fusions
- AMP: amplification
- HOMDEL: deep/homozygous deletion
- GAIN: copy number gain
- HETLOSS: shallow deletion / loss of heterozygosity
- CNA >= GAIN: equivalent to GAIN + AMP
- EXP > x or < -x: mRNA expression x SD above or below mean
- PROT > x or < -x: protein expression x SD above or below mean

# Modifiers:
- DRIVER: restrict to driver events
- GERMLINE / SOMATIC: restrict to mutation origin

# Operators:
- !=: exclude a specific mutation
- DATATYPES: apply keywords to multiple genes

# Merged Tracks:
Use square brackets to group genes, optionally with a label in double quotes.
Example: ["TP53 PATHWAY" TP53 P53AIP1]

# Notes:
- If given an Ensembl ID, convert it to a HUGO symbol before generating the query.
- Do NOT explain, format, or wrap in markdown—only return the raw OQL query.
- If a variant type is not specified in the message, use gene ID alone to build the query. 

Only respond with a complete, valid query. Do not include any commentary or explanation.

#example queries:
"show me genes in the MAPK pathway"
KRAS NRAS BRAF MAP2K1 MAP2K2 MAP3K1 MAP3K3 MAP3K7 RAF1 RPS6KA3

"query for all EGFR driver fusion events"
EGFR: FUSION_DRIVER

"query TP53 mutations except for missense mutations"
TP53: MUT != MISSENSE

"show me BRCA1 nonsense germline driver mutations"
BRCA1: NONSENSE_GERMLINE_DRIVER

"search for all KRAS mutations at position 12"
KRAS: MUT = (12-12)

"search for all KRAS mutations at positions 12 and 13"
KRAS: MUT = (12-13)
"""

class OQLGenerator:
    def __init__(
        self,
        model_name: str = "gpt-4.1",
    ) -> None:
        self.model_name = model_name
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
            model_name=os.getenv("OPENAI_MODEL"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

    def generate_oql(self, input_string):
        messages = [
            (
                "system",
                context_prompt,
            ),
            ("human",
             input_string),
        ]

        res = self.llm.invoke(messages)

        return res


# oql_generator = OQLGenerator()
# print(oql_generator.generate_oql('query for all EGFR driver fusion events'))
