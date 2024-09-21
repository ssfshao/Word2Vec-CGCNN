import pandas as pd
import os
import regex
import numpy as np
import math
from gensim.models import Word2Vec
from mat2vec.processing import MaterialsTextProcessor
import json

Path = './process/'
if os.path.exists((Path_cos:=Path+'cos')) == False:
    os.mkdir(Path_cos)

T_P = MaterialsTextProcessor()
Elements = T_P.ELEMENTS
Element_Names = T_P.ELEMENT_NAMES
N_M = ['H','He','B','C','N','O','F','Ne','Si','P','S','Cl','Ar','As','Se','Br','Kr','Te','I','Xe','At','Rn']
N_M_N = ['hydrogen','helium','boron','carbon','nitrogen','oxygen','fluorine','neon','silicon','phosphorus','sulfur','chlorine','argon','arsenic','selenium','bromine','krypton','tellurium','iodine','xenon','astatine','radon']
Delet = [element for element in Elements+Element_Names + [element_name.capitalize() for element_name in Element_Names] if element not in N_M + [element.lower() for element in N_M_N] + N_M_N]
Delet.append('fullerene')

Abstracts_Handle = list(pd.read_table(Path+'Abstracts_Handle.txt')['Abstracts_Handle'])
Data = [Abstract_Handle.split() for Abstract_Handle in Abstracts_Handle]
F_Data = pd.DataFrame({'Abstracts_Split':Data})
F_Data.to_csv(Path + 'Abstracts_Split.txt',sep='\n',index=False)

Model = Word2Vec(Data, vector_size = 200, sg=1, window = 8, negative = 15, alpha = 0.01,epochs = 5, min_count = 1,seed = 2023)
Model.save(Path + 'Word2vec.model')

Cems = pd.read_table(Path+'Cems.txt')['Cems'].values.tolist()
Cems_Word2vec = list(set(Cems).intersection(set(Model.wv.index_to_key)))

Metal_Ions = set(pd.read_table(Path+'Metal_Ion.txt')['Metal_Ion'].values.tolist())
Ions = set(filter(lambda T: T_P.ELEMENT_VALENCE_IN_PAR.match(T) != None, Cems_Word2vec))
Element_Ions = Ions.intersection(Metal_Ions)
F_Element_Ions = pd.DataFrame({'Element_Ions':list(Element_Ions)})
F_Element_Ions.to_csv(Path+'Element_Ions.txt',sep='\n',index=None)

Element_A = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
             'Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd',
             'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
             'Hf','Ta','W','Ir','Pt','Au'
            ]
Element_B = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
             'Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd',
             'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
             'Hf','Ta','W','Ir','Pt','Au'
            ]
AxByOz = regex.compile(r"^("+r"|".join(Element_A) + r")\d*" + r"("+r"|".join(Element_B)+r")\d*"+r"O\d*$")

Perovskite = list(filter(lambda T: AxByOz.match(T) != None, Cems_Word2vec))
F_Perovskite = pd.DataFrame({'Perovskite':Perovskite})
F_Perovskite.to_csv(Path+'Perovskite.txt',sep='\n',index=None)

def L(Name):
    if not os.path.exists(Path + 'Ligand'):
        os.mkdir(Path + 'Ligand')
    with open(Path + Name, 'r', encoding='utf-8') as file:
        Dict_Ligand = json.load(file)
    Cems_Word2vec_Lower = [cem.lower() for cem in Cems_Word2vec]
    Ligand={}
    for Key,Values in Dict_Ligand.items():
        if Key == 'Ligand/':
            continue
        All=[]
        for key,values in Dict_Ligand[Key].items():
            if key in ['COF_Covalent_Organic_Framework_Materials','MOF_finished_product']:
                continue
            All.append(pd.DataFrame({'ENs':values['Standard_EN'].values.tolist()}))
            All.append(pd.DataFrame({'ENs':values['Common_EN'].values.tolist()}))
        All_Concat = pd.concat(All).drop_duplicates().dropna()
        ENs = [cem.lower().replace(' ','_') for cem in All_Concat['ENs'].values.tolist()]
        L = list(set(ENs).intersection(set(Cems_Word2vec_Lower)))
        C,Cems=[],[]
        for Cem in Cems_Word2vec:
            if (cem:=Cem.lower()) in L and cem not in C:
                Cems.append(Cem)
                C.append(cem)   
        if Cems == []:
            continue
        Cems=list(filter(lambda T: regex.search('r'+'|'.join(Delet),T) == None, Cems))
        F = pd.DataFrame({Key:Cems})
        F.to_csv(Path+Key+'(Ligand).csv',sep=',',index=None,header=0)
        Ligand[Key.replace('Ligand/','')] = Cems
    return Ligand

Ligand = L('Ligand.json')

def build_complexes(As,Bs,Cs):
    Complexes = []
    for A in As:
        for B in Bs:
            for C in Cs:
                Complexes.append(A+' '+B+' '+C)
    return list(np.array(Complexes).flatten())

def C(Cs):
    Complexes=[]
    for cem in Perovskite:
        cem1 = regex.split(r'(?=[A-Z])', regex.sub(r'\d+','',cem))
        As = list(filter(lambda T: regex.search(cem1[1],T) !=None, Element_Ions))
        Bs = list(filter(lambda T: regex.search(cem1[2],T) !=None, Element_Ions))
        Complexes.append(build_complexes(As,Bs,Cs))
    return list(set(sum(Complexes,[])))

def build_vector(Complex):
    V_C = []
    Parts = Complex.split(' ')
    for i in range(len(Parts)):
        if i == 0:
            V_C = Model.wv[Parts[i]]
        else :
            V_C = np.add(V_C,Model.wv[Parts[i]])
    return V_C

def cos(array1, array2):
    norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array1))))
    norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array2))))
    return sum([array1[i]*array2[i] for i in range(0, len(array1))])/(norm1 * norm2)

def relevance_cos(L,KeyWord,Name):
    Vector_Key = Model.wv[KeyWord]
    R = [cos(build_vector(Complex),Vector_Key) for Complex in L]
    F = pd.DataFrame({'Cem':L,'Relevance':R}).sort_values(by='Relevance',ascending=False)
    if os.path.exists((Path_Name:=Path_Key+'/'+Name.split('/')[0])) == False:
        os.mkdir(Path_Name)
    F.to_csv(Path_Key+'/'+Name+'.csv',sep=',',index=None)
    return F

Heterojunction = ['van_der_waals','van_der_waals_heterojunction','vdw_heterojunction',
                  'schottky_heterojunction','janus_heterojunction','ohmic_heterojunction',
                  'p-p_heterojunction','p-n_heterojunction','n-n_heterojunction','n-p_heterojunction',
                  'p-n-p_heterojunction','n-p-n_heterojunction','quantum_heterojunction','type_I_heterojunction',
                  'type_II_heterojunction','homogeneous_heterojunction','heterogeneous_heterojunction',
                  'discontinuous_heterojunction','superlattice_heterojunction','osterwalder_heterojunction'
                 ]
Cems_Complexes={}
for Key,Values in Ligand.items():
    if os.path.exists((Path_Key:=Path_cos+'/'+Key)) == False:
        os.mkdir(Path_Key)
    Complexes = C(Values)
   
    if 'gas_sensor' in Cems_Word2vec:
        Perovskite_gas_sensor = relevance_cos(Complexes,'gas_sensor',Key+'_Gas_Sensor'+'/'+Key+'_Gas_Sensor')        
    if 'perovskite' in Cems_Word2vec:
        Perovskite_perovskite = relevance_cos(Complexes,'perovskite',Key+'_Perovskite'+'/'+Key+'_Perovskite')        
    if 'MOF' in Cems_Word2vec:
        Perovskite_MOF = relevance_cos(Complexes,'MOF',Key+'_MOF'+'/'+Key+'_MOF')        
    for c_s in ['triclinic','monoclinic','orthorhombic','trigonal','hexagonal','tetragonal','cubic']:
        if c_s in Cems_Word2vec:
            Cos_CS1 = relevance_cos(Complexes,c_s,Key+'_CrystalForm'+'/'+Key+'_'+c_s.capitalize())            
    for h_j in Heterojunction:
        if h_j in Model.wv.index_to_key:
            Cos_heterojunction1 = relevance_cos(Complexes,h_j,Key+'_Heterojunction'+'/'+Key+'_'+h_j)          

with open(Path + 'Cems_Complexes.json', 'w') as f:
    json.dump(Cems_Complexes, f, indent=4, ensure_ascii=False)