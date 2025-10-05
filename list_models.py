import pyiqa

models = pyiqa.list_models()
print(f'Total models available: {len(models)}')
print('\nAll models:')
for i, m in enumerate(models, 1):
    print(f'{i:2d}. {m}')

# Look for models that might be relevant for comparison
print('\n--- Models containing key terms ---')
key_terms = ['brisque', 'piqe', 'niqe', 'clip', 'unique', 'quality', 'assess']
for term in key_terms:
    relevant = [m for m in models if term in m.lower()]
    if relevant:
        print(f'{term.upper()}: {relevant}')