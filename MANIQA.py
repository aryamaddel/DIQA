import pyiqa

metric = pyiqa.create_metric("maniqa")
print(metric("koniq10k_512x384/826373.jpg").item())
