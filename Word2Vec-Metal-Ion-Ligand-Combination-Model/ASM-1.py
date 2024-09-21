import pandas as pd
import regex
import nltk
from chemdataextractor.doc import Paragraph
from mat2vec.processing import MaterialsTextProcessor

Path = './process/'

T_P = MaterialsTextProcessor()
Elements = T_P.ELEMENTS
Element_Names = T_P.ELEMENT_NAMES
Elements_All = Elements + [element.lower() for element in Elements] + Element_Names + [element_name.capitalize() for element_name in Element_Names]

T_M= ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
      'Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd',
      'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
      'Hf','Ta','W','Ir','Pt','Au'
     ]

N_M = ['H','He','B','C','N','O','F','Ne','Si','P','S','Cl','Ar','As','Se','Br','Kr','Te','I','Xe','At','Rn']
N_M_N = ['hydrogen','helium','boron','carbon','nitrogen','oxygen','fluorine','neon','silicon','phosphorus','sulfur','chlorine','argon','arsenic','selenium','bromine','krypton','tellurium','iodine','xenon','astatine','radon']

T_M_S = ['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Mc','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Tl','Po','Fr','Ra']

Elements_B = [element for element in Elements if element not in N_M+T_M_S]

Valence_Info = T_P.VALENCE_INFO
Units = T_P.SPLIT_UNITS + ['g-1','degrees','C','F','nl','Cm3','kg-1','W','mol','L-1','s-1','mM-1']

def metal_ion_standardization(Ts):
    Valence_State = ['','(I)','(II)','(III)','(IV)','(V)','(VI)','(VII)','(VIII)']
    M_E_N = {'M_E':[element.lower() for element in Elements if element not in N_M],
        'M_E_N':(M_E_N_L := [element_name for element_name in Element_Names if element_name not in N_M_N])
        }
    S = Sent = ' '.join(Ts)
    while (VS := Valence_Info.search(Sent)) != None and (M := regex.findall("[\u4e00-\u9fef]|[0-9.]*[0-9]+|[a-zA-Z]+\'*[a-z]*|.",Sent[:Sent.find(VS[0])],regex.UNICODE)) != []:
        VS_new = '('+VS[0].replace('(','').replace(')','')+')'
        if len(M) >=2 and M[-1] in [' ','-']:
            if M[-2].lower() in M_E_N['M_E']:
                S = S.replace(M[-2] + M[-1] + VS[0], M[-2].capitalize() + VS_new)
            elif M[-2].lower() in M_E_N['M_E_N']:
                S = S.replace(M[-2] + M[-1] + VS[0], M_E_N['M_E'][M_E_N['M_E_N'].index(M[-2].lower())].capitalize() + VS_new)
        else:
            if M[-1] in M_E_N['M_E']:
                S = S.replace(M[-1] + VS[0], M[-1].capitalize() + VS_new)
            elif M[-1].lower() in M_E_N['M_E_N']:
                S = S.replace(M[-1] + VS[0], M_E_N['M_E'][M_E_N['M_E_N'].index(M[-1].lower())].capitalize() + VS_new)
        Sent = Sent[Sent.index(VS[0]) + len(VS[0]):]    
    Sent = S
    while (VS := regex.search('\ ?\d\+|\ ?\(\d\+\)|\ ?\(\+\)|\ ?\+|\d\ \+',Sent)) != None and (M := regex.findall("[\u4e00-\u9fef]|[0-9.]*[0-9]+|[a-zA-Z]+\'*[a-z]*|.",Sent[:Sent.find(VS[0])],regex.UNICODE)) != []:
        if (N := regex.findall(r'\d+',VS[0])) == []:
            N = ['1']
        if int(N[0]) <= 7:
            if M[-1].lower() in M_E_N['M_E']:
                S = S.replace(M[-1]+VS[0], M[-1].capitalize()+Valence_State[int(N[0])])
            elif M[-1].lower() in M_E_N['M_E_N']:
                S = S.replace(M[-1]+VS[0], M_E_N['M_E'][M_E_N['M_E_N'].index(M[-1].lower())].capitalize()+Valence_State[int(N[0])])
        Sent = Sent[Sent.index(VS[0])+len(VS[0]):]
    return S.split()

def sup_standardization(Ts):
    for i in range(len(Ts)):
        while (T := regex.search('\([0-9]+\.{0,1}[0-9]{0,2}\)|\(x\)|\([0-9]+\.{0,1}[0-9]{0,2}[+-]x\)|\(x[+-][0-9]+\.{0,1}[0-9]{0,2}\)|\(\'\)',Ts[i])) != None:
            Ts[i] = Ts[i].replace(T[0],T[0].replace('(', '').replace(')', ''))
    return Ts

def bracket(Ts):
    Ts_Bracket = list(filter(lambda T: regex.search('^\[.*\]\[.*\]$',T) != None, Ts))
    [Ts[Ts.index(T_B)].replace('][','_').replace('[','').replace(']','') for T_B in Ts_Bracket]
    return Ts

def splicing_token(Ts):
    Sentence = ' '.join(Ts)
    if regex.search('\( \d \)',Sentence) != None:
        Sentence = Sentence.replace(regex.search('\( \d \)',Sentence)[0],regex.search('\( \d \)',Sentence)[0].replace(' ',''))
    for i in range(len(Ts)-1):
        if Ts[i] in ['–','-'] and Ts[i-1] != '<nUm>' and Ts[i+1] != '<nUm>':
            X_X = Ts[i-1],Ts[i],Ts[i+1]
            Sentence = Sentence.replace(' '.join(X_X),''.join(X_X))
        elif Ts[i] in ['@','/'] and Paragraph(Ts[i-1]).cems != None and Paragraph(Ts[i+1]).cems != None:
            X_X = Ts[i-1],Ts[i],Ts[i+1]
            Sentence = Sentence.replace(' '.join(X_X),''.join(X_X))
        elif Ts[i] == '<nUm>' and Ts[i+1] == 'x' and regex.search('10',Ts[i+2]):
            X_X = Ts[i],Ts[i+1],Ts[i+2]
            Sentence = Sentence.replace(' '.join(X_X),''.join(X_X))
        elif Ts[i] in Units and Ts[i+1] in Units:
            X_X = Ts[i],Ts[i+1]
            Sentence = Sentence.replace(' '.join(X_X),''.join(X_X))    
    
    Cems_All = []
    Abbrs = Paragraph(Sentence).abbreviation_definitions
    for Abbr in Abbrs:
        if Abbr[2] == 'CM' and len(Abbr[1]) == 1:
            Cems_All.append(Abbr[0][0])
            Cems_All.append(Abbr[1][0])
        elif Abbr[2] == 'CM' and len(Abbr[1]) != 1:
            if Abbr[1].count('-') != 0 or Abbr[1].count('/') != 0:
                for i in range(len(Abbr[1])-1):
                    if Abbr[1][i] == '-' or Abbr[1][i] == '/':
                        X_X = Abbr[1][i-1],Abbr[1][i],Abbr[1][i+1]
                        Abbr_Handle = ' '.join(Abbr[1]).replace(' '.join(X_X),''.join(X_X)).split()
                        Sentence = Sentence.replace(' '.join(Abbr_Handle),'_'.join(Abbr_Handle).replace('(_','(').replace('_)',')'))
                Cems_All.append(Abbr[0][0])
                Cems_All.append('_'.join(Abbr_Handle).replace('(_','(').replace('_)',')'))
            else:
                Sentence = Sentence.replace(' '.join(Abbr[1]),'_'.join(Abbr[1]).replace('(_','(').replace('_)',')'))
                Cems_All.append(Abbr[0][0])
                Cems_All.append('_'.join(Abbr[1]).replace('(_','(').replace('_)',')'))    
    
    Cems = [str(Cem) for Cem in Paragraph(Sentence).cems]
    for Cem in Cems:
        if regex.search('AdO|¬|…|‡|#|@|sup| aC|coordinate|tiv|gjTi|aU|vGjie|aA|site|aA|aOO|aE',Cem) != None:
            continue
        if len(Cem.split()) > 1:
            if Cem.find('—') != -1:
                Sentence = Sentence.replace(Cem,Cem.replace(' ',''))
                Cems_All.append(Cem.replace(' ',''))
            else:
                Sentence = Sentence.replace(Cem,Cem.replace(' ','_').replace('(_','(').replace('_)',')'))
                Cems_All.append(Cem.replace(' ','_').replace('(_','(').replace('_)',')'))
        else:
            Cems_All.append(Cem)
    
    Delete = ['<nUm>','<_sub_>','<nUm','nUm>',':','*','=','and','->','<_nUm_>','_or_','_to_','L_(III)','[Delta]','$']
    Cems_Delete = [Cem for Cem in Cems_All if list(filter(lambda delete: Cem.find(delete) != -1, Delete)) == []]
    
    return Sentence,Cems_Delete

def R(Abs):
    R = [('center dot ',''),
        (')and',') and'),
        ('gas sensor','gas_sensor'),
        ('( ','('),
        (' )',')'),
        (' · ','·'),
        ('¬',' ¬ '),
        ('Cu(RE)O2with','Cu(RE)O2 with'),
        ('Cu(I)(RE)O2with','Cu(I)(RE)O2 with'),
        ('Type','type'),('type I','type_I'),('type-I','type_I'),
        ('Schottky','schottky'),('van der waals','van_der_waals'),(' vdW ',' vdw '),
        ('van der Waals','van_der_waals'),('heterogeneous junction','heterojunction'),
        (' heterojunctions ',' heterojunction '),(' junctions ',' junction ')
    ]
    for r in R:
        Abs = Abs.replace(r[0],r[1])
    while (T1:=regex.search('(van_der_waals|vdw|schottky|janus|ohmic|p-p|p-n|n-n|n-p|p-n-p|n-p-n|quantum|homogeneous|heterogeneous|discontinuous|superlattice|osterwalder|type\_(I|II))\ (heterojunction|junction)',Abs)) != None:
        if T1[0].find('heterojunction') !=-1:
            Abs = Abs.replace(T1[0],T1[0].replace(' heterojunction','_heterojunction'))
        else:
            Abs = Abs.replace(T1[0],T1[0].replace(' junction','_heterojunction'))
    if (T2:=regex.search('\d\ \'',Abs)) != None:
        Abs = Abs.replace(T2[0],T2[0].replace(' ',''))
    return Abs

def token_handle(Ts):
    Tokens = metal_ion_standardization(Ts)
    Tokens = sup_standardization(Tokens)
    Tokens = bracket(Tokens)
    Tokens = splicing_token(Tokens)
    return Tokens[0],Tokens[1]

F_Abs1 = pd.read_table(Path + 'Abs1.txt',header = None, names = ['Abstract'])
F_Abs2 = pd.read_table(Path + 'Abs2.txt',header = 0, names = ['Abstract'])
F_Abs3 = pd.read_table(Path + 'Abs3.txt',header = None, names = ['Abstract'])
Abstracts = list(pd.concat([F_Abs1,F_Abs2,F_Abs3]).drop_duplicates().dropna()['Abstract'])

Abstracts_Handle, Cems, Abbrs, Data = [],[],[],[]
for Abstract in Abstracts:
    Abstract = R(Abstract)
    Handle_Result = token_handle(T_P.process(str(Abstract),normalize_materials=False)[0])
    [Cems.append(Cem) for Cem in Handle_Result[1] if Cem not in Cems]
    Abstracts_Handle.append(Handle_Result[0])

F_Abstracts_Handle = pd.DataFrame({'Abstracts_Handle':Abstracts_Handle})
F_Abstracts_Handle.to_csv(Path + 'Abstracts_Handle.txt',sep='\n',index=False)
F_Cems = pd.DataFrame({'Cems':Cems})
F_Cems.to_csv(Path+'Cems.txt',sep='\n',index=False)
