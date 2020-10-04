from itertools import product

feature_weighting_cartesian_prod = { 
    'text_high': [0.5, 1],
    'word_n_grams': [0, 0.5, 1],
    'char_n_grams': [0, 0.5, 1],
    'sentiment': [0, 0.5, 1],
    'opinion_towards': [0.5, 1],      
    'target': [0, 0.5, 1],               
}

feature_weightings_to_check = [dict(zip(feature_weighting_cartesian_prod, v)) for v in product(*feature_weighting_cartesian_prod.values()) if 
            dict(zip(feature_weighting_cartesian_prod, v)) != {'text_high': 0, 'word_n_grams': 0, 'char_n_grams': 0, 'sentiment': 0, 'opinion_towards': 0, 'target': 0}]

print(len(final))
