Below are the summaries which are generated when I ran 1% of the COVID- related data. I have also put the
computation of 10% data but its taking to long to execute because of the bottleneck at the summarize_clusters method.

This bottleneck is arriving due to the heavy computation matrix computation required for calculating the 
similarity_matrix. Since I am appending all the documents of a single cluster into a single document. It is
mounting up to a huge document which leads to huge number of sentences and upon vectorizing the document corpus
using Tfidfvectorization, its generating a huge matrix . I have tried to capture only the import features by
using dimensionality reduction(PCA) but its not that accurate as the one without using it.


Approach of generating the summary.

As we know the number of optimal clusters using the knee-elbow method and are grouping the documents into
respective clusters. The summarize_clusters method takes the dictionary containing the cluster_indicies along with
the document associated with it. I then traverse to each item in the dictionary take the cluster_index and the
document associated with it and pass it to summarize() method. Inside the summarize() method,first I generate
list of all sentences in document using parse_document method and then I normalize each
sentence to generate the vectorized_matrix using Tfidfvectorizer. Then, I use it to compute the
similarity_matrix. Using the networkxx package, I have generated and similarity_graph upon which pagerank algorithm 
is applied to get the rank of each sentence. Then I have sorted all the sentences_index based on their rank
and only taken the indices of top 10 sentences. Using the indices , I get the actual sentences and return the
list of top sentences back to the summarize_clusters() method which then appends the summarized result and 
the cluster_index to a list as a tuple. Then using the list of tuples, I traverse through each of the tuple 
and generate a SUMMARY-{cluster_index}.MD for each each cluster with the summarized result of the cluster.
The SUMMARY-{cluster_index}.MD is created inside a OUTPUT directory where the project is run.
   





######## SUMMARY- 0 ############

Taken together, these results suggest that transcriptional regulation of immune genes plays and important role in t
he immunity of channel catfish against bacterial infections.When the expression levels of the 27 genes at different
 time points after the first infection were compared to that after the second infection, significantly higher expre
ssion levels of these genes were induced by the first infection.Recently, He and coworkers (He et al., 2008 (He et 
al., , 2009 reported several cases of homologous recombination in influenza, their proposed recombination cases, ho
wever, suffer from the criticisms of 1) no phylogenetic support, 2) requiring parental sequences from substantially
 different geographic locations and time periods, 3) lone recombinant sequences and no circulating clades of recomb
inant viruses, and 4) unknown quality in sample-handling and sequencing (Boni et al., 2010) .7) , with significantl
y (P < 0.05) higher expression induced by the first infection at both time points.Serum lysozyme activity at variou
s time points after the first infection was significantly (P ¼ 0.005) higher than after the second infection (Fig.T
he PB2 protein has been shown to interact with the antiviral signaling protein (Graef et al., 2010) , affect host r
ange (Subbarao et al., 1993; Labadie et al., 2007) and virulence of influenza viruses (Hatta et al., 2001; Shinya e
t al., 2004) .Recently, HRV-C was identified based on molecular typing studies (Ledford et al., 2004; Lau et al., 2
007; Lee et al., 2007; McIntyre et al., 2013a Viruses had since revised the classification of HRV based on molecula
r typing results (Adams et al., 2013) .However, the expression levels of IL-1b was induced significantly (P < 0.05)
 higher by the first infection compared to that by the second infection (Fig.To date, there are only three peer-rev
iewed articles that have proposed the occurrence of intra-segmental homologous recombination in influenza A virus (
Gibbs et al., 2001; He et al., 2008 He et al., , 2009 ) and each study has been criticized and challenged (Worobey 
et al., 2002; Boni et al., 2010) .In the first published recombination study, Gibbs et al.The need for a simple met
hod of identifying HRV-C Despite delayed identification due to culture difficulties, studies suggest that HRV-C may
 cause more severe respiratory diseases than HRV-A or -B (McErlean et al., 2007; Miller et al., 2009; Bizzintino et
 al., 2011; Denlinger et al., 2011) .2017; Ma et al.Therefore, samples taken 7 to 10 days after infection may be ne
gative for virus by isolation in cell culture but positive by hybridization.


######## SUMMARY-1 ###########

All DHHC proteins autopalmitoylate (Fukata et al., 2004; Huang et al., 2004; Lobo et al., 2002; Mitchell et al., 20
06; Roth et al., 2002; Smotrys et al., 2005; Swarthout et al., 2005) , and autopalmitoylated PATs may represent cov
alent enzyme intermediates required for substrate palmitoylation (Hou et al., 2009) .Similar to neuronal synapses, 
many proteins in the TCR signaling pathway are palmitoylated and accumulate in lipid rafts, which coalesce and clus
ter at the IS following T cell activation (Bi et al., 2001; Burack et al., 2002) .Although it is clear that the pal
mitoylation status has a marked effect on lipid raft localization and function of T cell signaling molecules such a
s LAT, Src-family kinases (Lck and Fyn) and others, the importance of lipid raft localization is not fully understo
od.Although we do not yet know whether the selective hypopalmitoylation of LAT is the cause of T cell anergy, this 
is an intriguing possibility given the fact that downstream signaling events known to be impaired in anergic T cell
s such as Ras (Fields et al., 1996) , MAPK (Li et al., 1996) , NF-κB (Sundstedt et al., 1996) and AP-1 (Kang et al.
, 1992; Sundstedt and Dohlsten, 1998; Sundstedt et al., 1996) activation (Fathman and Lineberry, 2007; Schwartz, 20
03) , are dependent on intact LAT function (Finco et al., 1998; Lin et al., 1999; Ouellet et al., 2003) .In seeking
 the mechanism that underlies LAT hypopalmitoylation in anergic T cells, we considered an imbalance in the dynamic 
and reversible process of LAT palmitoylation/ depalmitoylation.Although it is clear that LAT palmitoylation is requ
ired for lipid raft localization (Lin et al., 1999; Zhang et al., 1998b) , the role and importance of LAT palmitoyl
ation in T cell development and activation has been a matter of some controversy.Indeed, recent studies demonstrate
d that T cells express two pools of LAT localized in the PM and the Golgi compartment and, furthermore, that mutati
on of Cys-26 and -29 leads to exclusive intracellular (Golgi) localization of LAT (Bonello et al., 2004; Hundt et a
l., 2009; Tanimura et al., 2003; Tanimura et al., 2006) .In fact, yeast 2-hybrid screens of PAT-interacting protein
s identified interactors that were later found to represent true substrates of the relevant DHHC proteins (Fernande
z-Hernando et al., 2006; Keller et al., 2004; Li et al., 2010; Nadolski and Linder, 2009; Saitoh et al., 2004; Sing
araja et al., 2002; Uemura et al., 2002) .Some PATs have protein-protein interaction domains Mitchell et al., 2006)
 , and regions in palmitoyl proteins distant from their palmitoylated Cys residues, which are required for substrat
e recognition and palmitoylation, have been identified; furthermore, swapping of these recognition sequences betwee
n distinct palmitoyl proteins can confer new substrate specificity patterns (Greaves et al., 2009; Huang et al., 20
09; Nadolski and Linder, 2009 ).A nonpalmitoylated PAG/Cbp mutant, which localized in the PM but not in lipid rafts
, was still tyrosine phosphorylated and associated with Csk, Fyn and EBP50 but, nevertheless, unlike wild-type PAG,
 it did not block proximal TCR signaling (Posevitz-Fejfar et al., 2008) .Given that LAT is not expressed in B cells
, a search for a LAT-like adaptor that may couple the antigen-specific B cell receptor (BCR) to downstream signalin
g pathways led to the discovery of non-T cell activation linker (NTAL) (Brdicka et al., 2002) , also known as linke
r for activation of B cells (LAB) (Janssen et al., 2003) , which, as implied by its name, is mainly expressed in no
n-T hematopoietic cells.Although CD4 and CD8αβ palmitoylation is not required for their transport or cell surface e
xpression this localization enhances, in the case of CD4, raft aggregation, clustering of the TCR and PKCθ at the I
S, and tyrosine phosphorylation of signaling proteins, primarily TCR-ζ and ZAP-70 kinase (Balamuth et al., 2004; Fr
agoso et al., 2003) .Earlier studies demonstrated defects in the activation of Ras (Fields et al., 1996) , MAPKs (L
i et al., 1996) , and the transcription factors NF-κB (Sundstedt et al., 1996) and AP-1 (Kang et al., 1992; Sundste
dt and Dohlsten, 1998; Sundstedt et al., 1996) in mouse and human anergic T cells, while TCR-induced Ca 2+ signalin
g remained relatively intact, the latter observation being consistent with an early report that treatment of primed
 T cells with a Ca 2+ ionophore can, in fact, induce T cell anergy (Jenkins et al., 1987) .




######### SUMMARY-2 #########

M. haemominutum infection was documented in 3 cats with IMHA over 4 studies (1 study showing no association with IM
HA),3, 51, 130, 140 yielding a median IME value of 2.16 and range of 0‐4.2.A high level of evidence was found that 
M. haemofelis causes IMHA in cats.Five studies documented 13 cases of IMHA in dogs infected with Babesia canis, wit
h a median IME value of 3.20 and range of 0‐4.32.57, 104, 111, 128, 134 Babesia vogeli was documented in 2 studies 
of 5 dogs with IMHA, with IME values of 5.73 and 4.14.Immune‐mediated hemolytic anemia was documented in 69 dogs in
 9 studies,93, 94, 95, 96, 99, 102, 103, 111, 151 with an additional study (in which the number of infected dogs wi
th IMHA could not be determined) providing useful mechanistic insight.125 The median IME value was 5.32, ranging fr
om 2.54 to 6.99.Four were studies of dogs experimentally infected with B. gibsoni, yielding a median IME value of 6
.41 and range of 6.08‐6.99.94, 95, 96, 99 Natural infection with B. gibsoni occurs most commonly in fighting breeds
.102, 125 However, mixed breed dogs used in experimental studies also develop IMHA, suggesting that the immune‐medi
ated pathogenesis is largely driven by the parasite.99 Whether other species of Babesia cause IMHA in dogs remains 
unclear.Six of the 9 cats were coinfected with feline leukemia virus (FeLV).138 Treatment for B. felis without immu
nosuppression resolved IMHA, yielding an IME value of 4.98.Although studies are limited, an intermediate level of e
vidence was found that B. felis causes IMHA in cats, and that treatment resolves IMHA without immunosuppression.Sev
en studies documenting IMHA in 21 cats infected with hemotropic Mycoplasma spp.The report of 2 of the cats with ery
throleukemia suggests that diagnosis was based on bone marrow cytological or histological review, or both.3 The met
hod of diagnosis of neoplasia was not described for the third cat with erythroleukemia.153 For the cats with non‐sp
ecified myeloproliferative disease, the diagnosis was based on bone marrow examination, but details are limited.110
 All cats with non‐specified myeloproliferative disease were FeLV positive.110 The evidence for a causal associatio
n between erythroleukemia and IMHA is negligible, with an IME value for the 1 study in which it could be assigned o
f 1.87.3 Other studies of erythroleukemia or unspecified myeloproliferarative disease did not yield IME values beca
use of the presence of comorbidities.A single cat with histiocytic sarcoma and IMHA was identified.3 This study did
 not show a causal association between IMHA and neoplasia, and the method of diagnosis of neoplasia was not describ
ed, yielding an IME value of 1.70.A single case of pancreatic carcinoma110 and a single case of anaplastic sarcoma3
 with giant cells in cats with IMHA were identified.The evidence for a causal association between IMHA and lymphoma
 is low, with a median IME value of 3.54 and a range of 1.87‐4.24.A single cat with multiple myeloma and IMHA was i
dentified.3 This study did not show a causal association between neoplasia and IMHA, and the method of diagnosis of
 neoplasia was unclear.For 3 additional studies, the number of dogs with Babesia and IMHA could not be determined.1
05, 125, 136 There is a high level of evidence that immune‐mediated destruction of erythrocytes contributes to anem
ia in dogs infected with B. gibsoni.The IME value for this study was 3.76.The evidence that A. phagocytophilum caus
es IMHA is low.Although no evidence for a causal link exists, cancer cannot be eliminated as a potential trigger fo
r this disease.Eight cats for which an IME value could be calculated for lymphoma and IMHA were identified.3, 9, 16
4 Two of the 3 studies reporting these cases did not demonstrate a causal association between IMHA and lymphoma,3, 
9 and 1 study was considered to partially report or suggest causality.164 The latter reports 2 sibling specific pat
hogen‐free experimental cats.


######## SUMMARY-3 ############

 outbreaks with little prior preparation,195 these moral lenses become important since they can help people develop
 an other-centric orientation and sensitivity to the needs of others.To systematically drive the importance of a pe
ople-centered approach to pandemic influenza, this chapter explicated the biological make-up of the influenza virus
 as well as the social and global features of the associated pandemic.But considering the limitations associated wi
th antiviral drugs as well as vaccines in relation to combating pandemic influenza, some form of non-therapeutic ap
proach is necessary, at least as some adjunct to mitigate the overall impact of pandemic influenza on the local and
 global human community.Responding to influenza vaccine safety signals during a pandemic constitutes a scientific a
nd public health policy issue since decision-makers must balance the immediate consequences of disease against unce
rtain risks.116 One of the consequences of the therapeutic uncertainties associated with pandemic influenza is the 
validity of administering potentially ineffective antiviral drugs with side-effects or vaccines that may cause harm
 to people.In addition, it is widely believed within the scientific community that influenza pandemics can hardly b
e halted, but they can be delayed.137 Therefore, the “ignorance gap” that occurs during pandemic influenza outbreak
s creates a context in which some of the preparatory strategies will inevitably fail (due to no fault of anyone), t
hereby leaving some people less protected.In relation to the socioeconomic dynamics, it is estimated that most infl
uenza pandemic-associated deaths occur in poor countries or in societies with scarce health resources which are alr
eady stretched by extant health priorities and challenges.138 Farmer and Campos underscore the need for bioethics t
o engage the growing problem posed by the gap between rich and poor nations, and how such a course of action reflec
ts social justice .139 Politically, communist nations such as China present unique dimensions to the vulnerabilitie
s of pandemic flu as they may control critical information traffic and access to patients, thereby deepening the cr
isis situation,140 or misrepresenting it, and thereby subjecting the rest of the connected world to avoidable risks
.The biological make-up of human beings both make them vulnerable to becoming infected with influenza virus as well
 as make them good vectors of dissemination.On that note, it showed how people-centered lenses such as communitaria
nism and ethics of care may be useful in engaging the associated practical and moral challenges.To clarify the impo
rtance of each of these approaches, the chapter elaborated each of these ethical lenses, and showed how each may he
lp orient different players in the context of a pandemic influenza towards acquiring a sense of community and an ot
her-centric sensitivity which will be essential to resolving the moral dilemmas as well as realizing the critical p
ublic health objective central to such a public health disaster .In fact, vaccines are hardly enough during pandemi
cs, and rationing is generally considered as the ethical option.120 Yet, the contemporary interconnection between h
ealth, the right to health and human rights121 implies that withholding vaccines from some people who might be pote
ntial victims of a pandemic outbreak may be a human rights violation.The importance of this last remark will become
 clearer against the backdrop of the social and global features of pandemic influenza outbreaks, a. theme addressed
 in the next section of this chapter.An influenza pandemic has the potential to cause more deaths and illnesses tha
n any other public health threat.34 Influenza pandemics are characterized by a widely varying number of deaths,35 a
nd each outbreak has always underscored this notion.It may likewise create the dispersal of a virus novel to other 
continents that have experienced typical outbreaks.Geographical location plays a major role in public health,53 and
 disasters including health disasters are unique in that each affected region of the world has different social, ec
onomic, and health backgrounds.54 As such, while there is a global spread, the nature of each local context and how
 it responds shapes pandemic influenza in some key ways.Besides sore arm and redness at the injection site as well 
as red eyes which have been reported in earlier vaccine trials,78 there has been some association between increased
 incidence of narcolepsy in children and the use of the ASO3-adjuvanted vaccine for pandemic H1N1 influenza in Scan
dinavian countries.79 In addition, anecdotal reports of fetal deaths occurring shortly after vaccination emerged in
 2009 and raised public health concerns about vaccine safety.80 Another shortcoming associated with vaccination gen
erally is vaccine failure,81 which often creates a false sense of protection in recipients while allowing the conti
nued spread of infection .82 In relation to pandemic influenza specifically, vaccine failure was recently reported

########### SUMMARY -4 ############

More than five different resistance genes were found in 15 of the 17 strains and one strain harboured 11 of the 18 genes detected.Intermediate susceptibility (8.5% of
 all strains) and resistance (40.7% of all strains) to TC were found among isolates of all three species.Results: The results of the 60 penicillin resistant isolates 
are given in the following table:The highest MIC value of 256 lg/mL was found for ERY in 25% of the resistant isolates.The target region was amplified in PCR using DN
A prepared from type strains as well as clinical isolates as templates.Methods: A total of 108 clinical strains of SPN were included in this study.Analysis of the iso
lates in the RAPD-PCR with primers previously used to type aspergilli revealed identity of the patient's isolates, whereas the reference strains were different.Result
s: The isolates were identified as: S. anginosus (53 strains), S. constellatus (five strains) and S. intermedius (one strain).In this study, the V1 and V2 signature s
equences of 60 type strains of 54 different streptococcal species were determined.Objectives: Previous data has shown using S. pneumoniae (SP) strains with raised Co-
a MICs that bacterial inoculum and strain MIC may impact significantly on the antibacterial effect measures (ABE).In 16 patients subsequent CSF testing was performed,
 six with positive and 10 with negative result of PCR.


########## SUMMARY -5 ############


Unlike most disjunctions, this one yields a definite conclusion: Jay is in Sweden.The theory accounts for the informality of arguments in science and daily life, wher
eas logic is notoriously of little help in analyzing them.Shared mental models are critical for collaboration and agreement before intervening in a system.There are t
wo models that an innovation in Health Security always needs to be cognizant of: The systems model that describes the wicked problem being addressed so that intervent
ions can be designed, executed and monitored; the systems that describe the enterprise as discussed above in Forms of Innovation so that you understand how the two sy
stems (and a larger system they are both part of) will interact.One last point about adopting a systems thinking approach and using systems and mental models for inno
vation is the ability to leverage systems archetypes [66].In the case of innovation, it is a systems thinking approach to value creation across business models, manag
ement models, organization models, operating models as well as the end products and services.It is keeping in mind and using all of the characteristics of innovation 
(forward thinking, new (or new use of) ideas, feasible, viable and valuable.For example, the iPod is often considered a disruptive product innovation, but the reality
 is that it was a totally new service-based business model, iTunes, that provided the success for the iPod and the eventual disruption of the media industry and the f
uture creation of the smart phone market (and associated industry disruption) via its foundation as an App Store.From a systems perspective, products and services are
 the lowest level elements of the system, the most easily changed and, though critical and important, have the least impact on altering the system’s (enterprise’s) ef
ficient and effective performance of its mission and achievement of its purpose in delivering value to its stakeholders.The organizational and managerial models provi
de the form of the enterprise including the structure (What the different parts of the operating model are and how they are put together) to the system (enterprise), 
clarity and specificity to the goal (mission, purpose, objectives, constraints) of the system (enterprise), the rules the operational level of the system (operational
 model of the enterprise) must follow, including the rules for changing all of the above.The organizational model is not just the structure of the organization, but t
he climates within it and the overall culture.Incorporating the potential (likelihood) of irrationality on the part of the stakeholders, as well as the futures resear
chers decision making [54].Lastly, foresight and futures researchers are encouraged to regularly assess how bias14 and logical fallacies15 may be entering their resea
rch before applying their conclusions into constructing future innovation planning scenarios.Intervening and innovation in the system associated with Global health se
curity requires the following:Recognizing systems and systems thinkingMental models and problem framingDivergent thinkingDesign and experimentation (real time and for
esight)Returning to Weick and Sutcliffe [1] ‘…unexpected events often audit our resilience, everything that was left unprepared becomes a complex problem, and every w
eakness comes rushing to the forefront’: the complex threat landscape associated with global health security matters requires a mindset that embraces innovation and c
reativity.10a).Systems thinking, approaching problems and asking how various elements within a system, which could be an ecosystem, an organization, or something more
 dispersed such as a health care system, influence one another, uses an iceberg model to describe the process.The Systems Thinker site has a good series of articles33
 on how to get your organization to adopt systems thinking.This addresses the need to have a common language of innovation as discussed earlier.There are many ways to
 develop Systems Thinking within the organization.A systems model will also help identify all the interrelationships from which unintended consequences and unintended
 degrees of side effects occur.The systems models will help align the enterprise (the value creation system) with its “customers” (the value consuming system) within 
the context of their shared environment (the systems of all stakeholders).Begin to have meetings around diagrams, models, pictures instead of reports and spreadsheets
.An excellent introduction to what a system is and how to think about it can be found at the Systems Thinker web site.16 It is beyond the scope of this chapter addres
s all the concepts and issues of the systems paradigm, general systems theory, systems thinking, sensemaking, complexity, systems innovation and other concepts discus
sed here in more depth.Break through innovation is most often both a cause and an effect of industry (enterprises dedicated to a particular value proposition) or mark
et (stakeholders with similar value needs and wants) consolidation.Disruptive product or service innovations, those which totally reinvent or create new industries or
 markets are usually the result of innovations at the higher levels of the system.Likewise, a common language about innovation, in particular recognize all potential 
stakeholders identified in future studies, helps avoid overly focusing on technological opportunities and immediate value creation to include less obvious social conc
erns, expectations and pressures that make up the wicked problems the innovation is attempting to address.Mental models are difficult to change and are one of the big
gest barriers for innovation as they determine the level of assimilation of new ideas find and the level and quality of effort that will be put into them.



########## SUMMARY - 6 ############


Chen et al., 2010 Chen et al., , 2014 Chen et al., , 2016 Han et al., 2013; Han and Yoo, 2014; Kim et al., 2010; Patel et al., 2010; Sagong and Lee, 2011; Song et 
al., 2010; Subramaniam et al., 2010; Sun et al., 2012a Sun et al., ,b, 2010 Wang et al., 2013a; Yoo et al., 2010) .3.1.Reactivation of memory CD8+ T cells to effector
 CD8+ T cells is dependent on the presence of type I IFNs Furthermore, type I IFNs can also protect T cells from killing by NK cell by expressing NCR1 (natural cytoto
xicity triggering receptor 1; CD335) which is the negative signal for NK cells (Crouse et al., 2014; Xu et al., 2014) .The T cell response in pigs to PRRSV is crucial
 for specific anti-PRRSV activities directly or indirectly.Indeed, the induction of IFN production in vivo has resulted in the increase of adaptive immune responses (
Wang et al., 2013c) .Nsp1b inhibits both type I IFN production and signaling potently Chen et al., 2010; Patel et al., 2010; Wang et al., 2013b) .Other reports also s
how the regulation of type I IFNs on B cell activation, antibody secretion, and isotype switching during viral infections, suggesting that type I IFNs regulate the B 
cells activity generally and deeply in mice (Bach et al., 2007; Fink et al., 2006; Purtha et al., 2008) .The poor and delayed B cell-mediated specific antibody respon
se is one of the major obstacles to control PRRSV in pigs.During LCMV and vesicular stomatitis virus (VSV) infections in mice, type I IFNs contribute to the activatio
n and longer survival of CD8+ T cells and a larger number of clonal expansion (Aichele et al., 2006; Curtsinger et al., 2005; Keppler et al., 2012; Kolumam et al., 20
05; Le Bon et al., 2006a Marrack et al., 1999; Urban et al., 2016) .During influenza virus infection of mice, IFNAR on B cells is necessary for B cell activation, sug
gesting the role of IFNs for B cell activation in response to influenza virus infection (Chang et al., 2007; Coro et al., 2006; Rau et al., 2009) .Since pDCs are not 
permissive for PRRSV (Calzada-Nova et al., 2011) , the failure of pDC activation during PRRSV infection may be due to the lack of type I IFN production, suggesting a 
role of type I IFN for priming the adaptive response against PRRSV.Type I IFNs enhance the survival and function of NK cells directly or indirectly in murine models a
nd human cells ( Fig.The importance of NendoU activity in viral RNA synthesis and virus replication has been shown for EAV and SARS-CoV (Nedialkova et al., 2009; Post
huma et al., 2006) .PRRSV nsp11 is an IFN antagonist and the endoribonuclease activity is critical for IFN suppression when stimulated by poly(I:C) (Shi et al., 2011)
 .A nuclease motif has been identified at K18-E32 in the N-terminal region of nsp1b (Xue et al., 2010) , but this motif is irrelevant to IFN suppression (Han et al., 
2014) .Although PLP1b loses its activity once cleaving and releasing nsp1b from nsp2 (Sun et al., 2009) , it still plays crucial roles in nsp2TF gene expression, type
 I IFNs suppression, and viral replication.Additional studies unveil that type I IFNs can enhance the cytotoxicity of CD8+ T cells in mice (Agarwal et al., 2009; Curt
singer et al., 2005; Jennings et al., 2014; Marshall et al., 2010).


########### SUMMARY -7 ###########

Radiographs of the spine may be useful to detect lytic lesions, which are more likely to be associated with osteosarcoma (vertebral lymphosarcoma is rarely detectable
 radiographically).Nutritional hyperparathyroidism should be considered as an infrequent cause of spinal cord disease in cats, 18 Spinal trauma causes peracute/acute 
neurological signs, which usually remain static (nonprogressive) or improve over time; however, in the first 24-72 h worsening of the neurological signs can be notice
d due to secondary spinal cord hemorrhage/ischemia or edema.Lateralization of neurological signs is reported, depending on the lesion site.The most common neoplasm af
fecting the spinal cord of cats is lymphosarcoma, followed by vertebral column neoplasia secondarily affecting the spinal cord, 12-15 meningioma, 14, 16 and glial cel
l tumours 14, 17 ( Table 2) .Age seems to be associated with the prevalence of certain tumor types: cats younger than 5 years of age are significantly more likely to 
be affected by lymphosarcoma, usually with a shorter duration of neurological signs and multiple spinal cord segment involvement.36 Spinal trauma is an important caus
e of spinal cord dysfunction in cats.8, 9 Neurological signs usually relate to a focal lesion, which might be seen on spinal radiographs; in a traumatised cat, howeve
r, there may be multiple lesions affecting the spine and/or spinal cord, including ischemia, compression, hemorrhage and laceration.Congenital malformations, which ca
n be symmetrical or asymmetrical, should be one of the differentials for a kitten/ young cat.Metabolic diseases causing neurological deficits compatible with spinal c
ord localizations are rare.There are no idiopathic diseases of the spinal cord recognized in cats.Typically, neoplastic conditions are chronic progressive diseases, a
lthough acute and subacute onsets of neurological signs and/ or spinal pain are relatively frequent.✜ Spinal cord localization The four segments into which the spinal
 cord is clinically divided (C1-C5, C6-T2, T3-L3, L4-S3) are responsible for specific neurological signs when affected by disease processes (Table 1 ).Corticosteroid 
administration decreases the white blood cell count and/or total protein levels; 22 therefore, evaluation of CSF in pretreated cats may give false negative results.Fi
brocartilaginous embolic myelopathy Vascular disease, although a relatively uncommon disease type per se, 30,31 is the fourth most common cause of spinal cord dysfunc
tion in cats.23Lymphosarcoma has historically been considered to be the most common neoplasm affecting the spinal cord of cats and this has recently been reaffirmed b
y two large studies of cats with different spinal cord diseases.And cats with spinal cord pain were four times more likely to have abnormal MRI results than cats with
out spinal pain.30, 31 Cats with fibrocartilaginous embolic myelopathy typically present with a peracute/acute onset of asymmetrical spinal cord-related signs, which 
may progress over the first 24 h. 30 Spinal palpation does not elicit pain, and any spinal cord segment may be involved, although the intumescences are most frequentl
y affected.

######### SUMMARY -8 ############

TNF/CHP nanoparticles seemed to stimulate DCs most in the nasal mucosal immune tissue (Figure 6).To understand the nasal tissue activation caused by the TNF/CHP nanop
articles soon after immunization, we conducted gene expression profiling in NALT cells 6 h after nasal immunization with the IVV antigen combined with or without the 
nanoparticles.No obvious immunotoxicity was detected and the results suggested that TNF/CHP nanoparticles are relatively safe as a nasal vaccine adjuvant.The results 
of this study demonstrate that TNF/CHP nanoparticles are effective as a vaccine adjuvant when administered via the nasal mucosal route.The results show that the nasal
ly administered TNF/CHP nanoparticles combined with IVV induced significant levels of IgA in the nasal wash, as well as IgG1 in blood plasma (Figures 2(a) and 2(b)).O
verall, our data demonstrate that TNF/CHP nanoparticles are effective as a vaccine adjuvant for nasally delivered IVV.Being focused on the nasal route of vaccination,
 we examined immune cells in the nasal tissues after the immunization.The TNF/CHP nanoparticles enhanced host immunity and the effect seemed stronger than that of CTB
, a vaccine adjuvant positive control.To directly address the stimulatory effect of the TNF/CHP nanoparticles on protective immunity as a vaccine adjuvant, we carried
 out an experimental lethal influenza virus challenge of immunized mice.The effects of the TNF/CHP nanoparticles seemed stronger than those of CTB.To directly address
 the use of TNF/CHP nanoparticles as a vaccine adjuvant on protective immunity, we employed a lethal influenza challenge mouse model.When administered the IVV, TNF/CH
P nanoparticles induced high levels of IgA in the nasal wash, as well as IgG1 in blood plasma (Figures 2(a) and 2(b)).The resulting Alexa-labeled OVA was used as an i
ndicator antigen for nasal administration to mice with or without the TNF/CHP nanoparticles as an adjuvant.These data suggest that TNF/CHP nanoparticles have the pote
ntial as a vaccine adjuvant with a broad range of applications.TNF/CHP nanoparticles elicited immune activation comparable to that of CTB (Figure 3(a)).There were no 
TNF/CHP nanoparticles and IVV-related changes observed (data not shown).The anti-IVV IgG1 in blood was examined after the administration of TNF/CHP nanoparticles and 
IVV.

######### SUMMARY-9 #############

However, effectively inferring phylogenetic trees from the genome sequences and comparing the branching orders with taxonomy at all ranks present challenges.In the CV
Tree server, each built-in genome is associated with initial lineage information taken from the NCBI Taxonomy (www.ncbi.nlm.nih.gov/taxonomy).By expanding the nodes o
r making enquiry for a designated taxon name, any part of the tree may be unfolded for in-depth inspection.Before describing the applications of CVTree3, we recall th
e significant fact that for prokaryotes with sequenced genomes, all taxonomic revisions or new proposals published thus far agree with CVTree or at least do not contr
adict the CVTree branching order.Note that summing up the number of genomes shown explicitly in this figure yields 3219, because one genome used as out-group was hidd
en.A prominent feature of CVTree is its high resolution at the species level and below (for infrasubspecific ranks, see page 30, volume 2 of the Manual), far surpassi
ng the capability of 16S rRNA analysis.(6) The server allows print-quality output of any selected subtree.Since there is a detailed online (and printable) User’s Manu
al (File S1) for the web server, we will not describe in detail the aforementioned technical points except to demonstrate some useful features regarding the taxonomic
 placement of a few newly-sequenced genomes without proper lineage information given at the present time.Inherited from the previous releases, the CVTree3 web server 
has a built-in genome dataset.Lineage information containing one or more “Unclassified” terms is considered incomplete.4.A central notion in comparing tree branching 
orders with taxonomy is monophyly.A branch may converge to a monophyletic branch such as Cyanobacteria{77} or converge to several partially monophyletic clusters such
 as Clostridium{32/49}, Clostridium{7/49}, and Clostridium{4/49}, meaning that the 49 genomes listed under the genus Clostridium in the reference taxonomy appear as s
everal clusters in CVTree.5.There are two elements of a phylogenetic tree: the branching order (topology) and the branch lengths.These include the following: large-sc
ale classification, taxonomic placement of newly-sequenced genomes, and high resolution of CVTree at the rank species and below.Large-scale classification, or as Cava
lier-Smith puts it [56], mega-classification, of prokaryotes, deals with higher taxonomic ranks such as phylum, class, and order (at present, ranks higher than order 
are not covered by the International Bacterial Code [57]).The powerful and parallelized CVTree3 web server will bring about many additional new applications.Upon ente
ring the Working Project, a maximally-collapsed CVTree with three branches, corresponding to the three main domains of life, appears as shown in Figure 1.(2) The CVTr
ee3 web server is not only designed as a phylogenetic tool, but also enables combined study of phylogeny and taxonomy both on a large scale across many phyla and at l
ower ranks down to infrasubspecific strains.






