import codecs
with codecs.open('notebooks/demo.ipynb', 'r', 'utf-8-sig') as f:
    content = f.read()
with codecs.open('notebooks/demo.ipynb', 'w', 'utf-8') as f:
    f.write(content)
print('Fixed!')