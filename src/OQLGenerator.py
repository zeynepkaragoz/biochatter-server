from langchain_openai import AzureChatOpenAI
import os

context_prompt = """
OQL Keywords
MUT: All non-synonymous mutations
MUT = <protein change>: Specific amino acid changes (e.g. V600E or V600)
MUT = <mutation type>: Acceptable values are MISSENSE, NONSENSE, NONSTART, NONSTOP, FRAMESHIFT, INFRAME, SPLICE, TRUNC for each mutation type
FUSION: All fusions or all structural variants
AMP: Amplifications
HOMDEL: Deep or homozygous deletions 
GAIN: Gains
HETLOSS: Shallow deletions or loss of heterozygosity
Comparison operators can also be used with CNA (e.g. CNA >= GAIN is the same as AMP GAIN)
EXP < -x: mRNA expression is less than x standard deviations (SD) below the mean
EXP > x: mRNA expression is greater than x SD above the mean
The comparison operators <= and >= also work
PROT < -x: Protein expression is less than x standard deviations (SD) below the mean
PROT > x: Protein expression is greater than x SD above the mean
The comparison operators <= and >= also work
OQL modifiers
DRIVER: Include only mutations, fusions and copy number alterations which are driver events, as defined in OncoPrint (default: OncoKB and CancerHotspots).
GERMLINE: Include only mutations that are defined as germline events by the study.
SOMATIC: Include all mutations that are not defined as germline.
(a-b): Include all mutations that overlap with the protein position range a-b, where a and b are integers. If you add a * (i.e. (a-b*)) then it will only include those mutations that are fully contained inside a-b. The open-ended ranges (a-) and (-b) are also allowed.
When querying a gene without providing any OQL specifications, cBioPortal will default to these OQL terms for a query with Mutation and Copy Number selected in the Genomic Profiles section: MUT FUSION AMP HOMDEL
Proper formatting for OQL is straightforward: gene name, followed by a colon, followed by any OQL keywords and ending in a semicolon, an end-of-line, or both.
GENE1: OQL KEYWORDS;
GENE2: OQL KEYWORDS
alanine - ala - A 
arginine - arg - R 
asparagine - asn - N 
aspartic acid - asp - D 
cysteine - cys - C 
glutamine - gln - Q 
glutamic acid - glu - E 
glycine - gly - G 
histidine - his - H 
isoleucine - ile - I 
leucine - leu - L 
lysine - lys - K 
methionine - met - M 
phenylalanine - phe - F 
proline - pro - P 
serine - ser - S 
threonine - thr - T 
tryptophan - trp - W 
tyrosine - tyr - Y 
valine - val - V

OQL Documentation:
Basic Usage

When querying a gene without providing any OQL specifications, cBioPortal will default to these OQL terms for a query with Mutation and Copy Number selected in the Genomic Profiles section: MUT FUSION AMP HOMDEL

image of basic query
image of basic query

You can see the OQL terms applied by hovering over the gene name in OncoPrint:

image of basic query oncoprint
image of basic query oncoprint

If you select RNA and/or Protein in the "Genomic Profiles" section of the query, the default settings are:

RNA: EXP >= 2 EXP <= -2

Protein: PROT >= 2 PROT <= -2

image of exp prot query oncoprint
image of exp prot query oncoprint

You must select the relevant Genomic Profile in order for OQL to query that data type. For example, you can't add EXP > 2 to the query without also selecting an RNA profile.

Proper formatting for OQL is straightforward: gene name, followed by a colon, followed by any OQL keywords and ending in a semicolon, an end-of-line, or both.

GENE1: OQL KEYWORDS;
GENE2: OQL KEYWORDS

In general, any combination of OQL keywords and/or expressions can annotate any gene, and the order of the keywords is immaterial.

Below we will go into greater detail about each data type.
#
Mutations

To view cases with specific mutations, provide the specific amino acid change of interest:

BRAF: MUT = V600E

You can also view all mutations at a particular position:

BRAF: MUT = V600

Or all mutations of a specific type:

TP53: MUT = <mutation type>

<mutation type> can be one or more of:

    MISSENSE
    NONSENSE
    NONSTART
    NONSTOP
    FRAMESHIFT
    INFRAME
    SPLICE
    TRUNC

For example, to view TP53 truncating mutations and in-frame insertions/deletions:

TP53: MUT = TRUNC INFRAME

OQL for mutations can also be written without MUT =. The following examples are identical:

BRAF: MUT = V600E
BRAF: V600E
TP53: MUT = TRUNC INFRAME
TP53: TRUNC INFRAME

OQL can also be used to exclude a specific protein change, position or type of mutation. For example, below are examples to query all EGFR mutations except T790M, all BRAF mutations except those at V600 and all TP53 mutations except missense:

EGFR: MUT != T790M
BRAF: MUT != V600
TP53: MUT != MISSENSE

Note that this will only work to exclude a single event. Because OQL uses 'OR' logic, excluding multiple mutations or excluding a mutation while including another mutation (e.g. BRAF: MUT=V600 MUT!=V600E) will result in querying all mutations.
#
Copy Number Alterations

To view cases with specific copy number alterations, provide the appropriate keywords for the copy number alterations of interest. For example, to see amplifications:

CCNE1: AMP

Or amplified and gained cases:

CCNE1: CNA >= GAIN

Which can also be written as:

CCNE1: GAIN AMP

#
Expression

High or low mRNA expression of a gene is determined by the number of standard deviations (SD) from the mean. For example, to see cases where mRNA for CCNE1 is greater than 3 SD above the mean:

CCNE1: EXP > 3

#
Protein

High or low protein expression is similarly determined by the number of SD from the mean. For example, to see cases where protein expression is 2 SD above the mean:

EGFR: PROT > 2

Protein expression can also be queried at the phospho-protein level:

EGFR_PY992: PROT > 2

#
Modifiers

Modifiers can be used on their own or in combination with other OQL terms for mutations, fusions and copy number alterations to further refine the query. Modifiers can be combined with other OQL terms using an underscore. The order in which terms are combined is immaterial.
#
Driver

The DRIVER modifier applies to mutations, fusions and copy number alterations. The definition of what qualifies as a driver alteration comes from the "Mutation Color" menu in OncoPrint. By default, drivers are defined as mutations, fusions and copy number alterations in OncoKB or CancerHotspots.

On its own, the DRIVER modifier includes driver mutations, fusions and copy number alterations:

EGFR: DRIVER

Or it can be used in combination with another OQL term. For example, to see only driver fusion events:

EGFR: FUSION_DRIVER

Or driver missense mutations:

EGFR: MUT = MISSENSE_DRIVER

When combining DRIVER with another OQL term, the order doesn't matter: MUT_DRIVER and DRIVER_MUT are equivalent. DRIVER can be combined with:

    MUT
    MUT = <mutation type> or MUT = <protein change>
    FUSION
    CNA
    AMP or GAIN or HETLOSS or HOMDEL
    GERMLINE or SOMATIC (see below)

#
Germline/Somatic

The GERMLINE and SOMATIC modifiers only apply to mutations. A mutation can be explicitly defined as germline during the data curation process. Note that very few studies on the public cBioPortal contain germline data.

GERMLINE or SOMATIC can be combined with:

    MUT
    MUT = <mutation type> or MUT = <protein change>
    DRIVER

To see all germline BRCA1 mutations:

BRCA1: GERMLINE

Or to see specifically truncating germline mutations:

BRCA1: TRUNC_GERMLINE
BRCA1: GERMLINE_TRUNC

The order is immaterial; both options produce identical results.

Or to see somatic missense mutations:

BRCA1: MUT = MISSENSE_SOMATIC

GERMLINE or SOMATIC can also be combined with DRIVER and, optionally, a more specific mutation term (e.g. NONSENSE):

BRCA1: NONSENSE_GERMLINE_DRIVER

#
The DATATYPES Command

To save copying and pasting, the DATATYPES command sets the genetic annotation for all subsequent genes. Thus,

DATATYPES: AMP GAIN HOMDEL EXP > 1.5 EXP < -1.5; CDKN2A MDM2 TP53

is equivalent to:

CDKN2A: AMP GAIN HOMDEL EXP > 1.5 EXP < -1.5
MDM2: AMP GAIN HOMDEL EXP > 1.5 EXP < -1.5
TP53: AMP GAIN HOMDEL EXP > 1.5 EXP < -1.5

#
Merged Gene Tracks

OQL can be used to create a merged gene track in OncoPrint, in which alterations in multiple genes appear as a single track. This is done by enclosing a list of genes in square brackets. By default, the track will be labeled by the gene names, separated by '/'. To instead specify a label, type the desired label within double quotes at the beginning of the square brackets. For example:

["CDK INHIBITORS" CDKN2A CDKN2B]
[MDM2 MDM4]

Please generate a OQL query that can be used for cbioportal, do use the keywords where possible, only respond with an OQL query. If an ensembl id is provided use the hugo symbols.
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
