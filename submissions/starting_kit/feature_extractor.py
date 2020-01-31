import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

class FeatureExtractor(object):
    
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        
        def conversion_code_voie(code):
            if code[0] == 'A':
                return 1
            elif code[0] > 'A' and code[0] < 'X':
                return 2
            elif code[0] == 'X':
                return 3
            elif code[0] == 'Y' or code[0] == 'Z':
                return 4
            else:
                return 0
   
        def process_surface_lot_total(X):
            surface_lot_total = X[["surface_lot_1","surface_lot_2","surface_lot_3","surface_lot_4","surface_lot_51"]].mean(axis = 1)
            surface_lot_total = surface_lot_total*X.nombre_lots
            return surface_lot_total.values.reshape(-1, 1)
        surface_lot_total_transformer = FunctionTransformer(process_surface_lot_total, validate=False)

        def process_date_mutation(X):
            date = pd.to_datetime(X.date_mutation, format='%Y-%m-%d')
            return np.c_[date.dt.year, date.dt.month]
        date_mutation_transformer = FunctionTransformer(process_date_mutation, validate=False)

        def process_nature_mutation(X):
            conversion_nature_mutation = {j:i for i,j in enumerate(set(X.nature_mutation))}
            nature_mutation = X.nature_mutation.map(conversion_nature_mutation)
            return nature_mutation[:, np.newaxis]
        nature_mutation_transformer = FunctionTransformer(process_nature_mutation, validate=False)

        def process_code_voie(X):
            code_voie = X.code_voie.copy()
            code_voie.loc[code_voie.notna()] = X.loc[X.code_voie.notna(), 'code_voie'].map(conversion_code_voie)
            return code_voie[:, np.newaxis]
        code_voie_transformer = FunctionTransformer(process_code_voie, validate=False)

        def process_suffixe_numero(X):
            l = np.sort(list(set(X.suffixe_numero)))
            conv = {l[i]:i for i in range(len(l))}
            conv['-'] = np.nan
            conv['.'] = np.nan
            conv['nan'] = np.nan
            suffixe = X.suffixe_numero.map(conv)
            suffixe = pd.to_numeric(suffixe, errors='coerce')
            return suffixe[:, np.newaxis]
        suffixe_numero_transformer = FunctionTransformer(process_suffixe_numero, validate=False)

        def numeric(X):
            cols = ['code_commune', 'code_departement', 'code_postal', 'code_type_local',
                    'nombre_lots', 'nombre_pieces_principales', 'numero_disposition', 
                    'numero_volume', 'prefixe_section', 'surface_relle_bati', 'surface_terrain']
            num_cols_ = []
            for col in cols:
                num_cols_ += [pd.to_numeric(X.loc[:,col], errors='coerce')]
            return pd.concat(num_cols_, axis=1)
        numeric_transformer = FunctionTransformer(numeric, validate=False)


        surface_totale_col = ['surface_lot_1','surface_lot_2','surface_lot_3','surface_lot_4','surface_lot_51', 'nombre_lots']
        date_mutation_col = ['date_mutation']
        nature_mutation_col = ['nature_mutation']
        code_voie_col = ['code_voie']
        suffixe_numero_col = ['suffixe_numero']

        num_cols = ['code_commune', 'code_departement', 'code_postal', 'code_type_local',
                    'nombre_lots', 'nombre_pieces_principales', 'numero_disposition', 
                    'numero_volume', 'prefixe_section', 'surface_relle_bati', 'surface_terrain']

        drop_cols = ['code_service_ch', 'reference_document', 'articles_1', 'articles_2' ,
                     'articles_3', 'articles_4', 'articles_5', 'identifiant_local', 'type_voie',
                     'section','numero_plan','nature_culture_speciale', 'nature_culture',
                     'surface_lot_1','surface_lot_2','surface_lot_3','surface_lot_4','surface_lot_51',
                     'type_local', 'commune', 'voie', 'lot_1', 'lot_2', 'lot_3', 'lot_4', 'lot_5']

        preprocessor = ColumnTransformer(
            transformers=[
                ('surface totale', make_pipeline(surface_lot_total_transformer, SimpleImputer(strategy='median')), surface_totale_col),
                ('date_mutation', make_pipeline(date_mutation_transformer, SimpleImputer(strategy='median')), date_mutation_col),
                ('nature_mutation', make_pipeline(nature_mutation_transformer, SimpleImputer(strategy='median')), nature_mutation_col),
                ('code_voie', make_pipeline(code_voie_transformer, SimpleImputer(strategy='median')), code_voie_col),
                ('suffixe_numero', make_pipeline(suffixe_numero_transformer, SimpleImputer(strategy='median')), suffixe_numero_col),
                ('num', make_pipeline(numeric_transformer, SimpleImputer(strategy='median')), num_cols),
                ('drop cols', 'drop', drop_cols),
            ])
        

        self.preprocessor = preprocessor
        self.preprocessor.fit(X_df, y_array)
        return self

    def transform(self, X_df):
        return self.preprocessor.transform(X_df)